"""
Implicit neural shapes (notebooks/rockerarm.ipynb, notebooks/bunny.ipynb, Section 4.3): fit f = 0 to the vertices
of a mesh with their normals, regularized by an eikonal term. Settings as for the paper figures (commits cb574c0/5ee82bc).
"""

import numpy as np
import torch
import trimesh
from models.model_architecture import GeneralNet
from training.optimizers import GaussNewton, GaussNewtonNew
from training.residuals import ResidualTerm
from util.surface_sampling import sample_model_surface_binsearch
from util.visualization.utils_mesh import get_mesh
from experiments.metrics import project_to_model, chamfer, rms, maximum, largest_component

NOTEBOOKS_DIR = __file__.rsplit("experiments", 1)[0] + "notebooks/"


class ImplicitShapeTask:
    loss_keys = ["data", "normal"]
    eikonal_weights = {"gn": 1e-2, "adam": 1e-1, "lbfgs": 1e-1}
    opt_defaults = {
        "gn":    dict(lr=1e-1, regularization=1e-6, do_line_search=True),
        "adam":  dict(lr=1e-3),  ## the cosine schedule of the notebooks was never stepped
        "lbfgs": dict(line_search_fn="strong_wolfe"),
    }
    gn_class = GaussNewtonNew  ## its line search tries all step sizes from the same point
    figure_opts = ["adam", "lbfgs", "gn"]
    ## Spurious components away from the data (seed dependent) are left out of the figures and the model-side metrics;
    ## the unfiltered metrics are kept with the suffix _all
    largest_component_only = True
    table_keys = ["loss", "cd", "cd_t2m", "cd_m2t", "hausdorff", "interface_rms", "normal_consistency", "eikonal_rms",
                  "cd_all", "hausdorff_all", "removed_area_fraction", "over_hole_fraction"]
    pretrain_extent = 1.5

    def __init__(self, subsample_fraction=1.0):
        mesh = trimesh.load(NOTEBOOKS_DIR + self.mesh_file)
        ## Normalize as in the notebooks: centered, max vertex norm 1
        center = mesh.vertices.mean(axis=0)
        scale = np.linalg.norm(mesh.vertices - center, axis=1).max()
        mesh.apply_translation(-center)
        mesh.apply_scale(1 / scale)
        self.mesh = mesh
        ## Open boundary edges (holes in the scan, e.g. the base of the bunny); empty for watertight meshes
        rim = mesh.edges_sorted[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
        self.rim_a, self.rim_b = mesh.vertices[rim[:, 0]], mesh.vertices[rim[:, 1]]
        pts = torch.tensor(mesh.vertices, dtype=torch.float64)
        normals = torch.tensor(mesh.vertex_normals, dtype=torch.float64)
        ## Subsample with a fixed generator, so all seeds and optimizers see the same data; the rest is held out
        perm = torch.randperm(len(pts), generator=torch.Generator().manual_seed(0), device="cpu").to(pts.device)
        n_sub = int(subsample_fraction * len(pts))
        self.pts, self.normals = pts[perm[:n_sub]], normals[perm[:n_sub]]
        self.pts_held_out = pts[perm[n_sub:]]
        self.box_min, self.box_max = pts.min(0).values - 0.2, pts.max(0).values + 0.2
        self.mesh_bbox = (tuple((pts.min(0).values - 0.05).tolist()), tuple((pts.max(0).values + 0.05).tolist()))
        ## Fixed evaluation samples
        self.pts_surface_true = torch.tensor(trimesh.sample.sample_surface(mesh, 20_000, seed=0)[0], dtype=torch.float64)
        self.pts_eval_eikonal = self.uniform_in_box(20_000, torch.Generator(device=pts.device).manual_seed(1))

    def uniform_in_box(self, n, generator=None):
        u = torch.rand([n, 3], dtype=torch.float64, generator=generator)
        return u * (self.box_max - self.box_min) + self.box_min

    @property
    def pretrain_bounds(self):
        return torch.tensor([[-self.pretrain_extent, self.pretrain_extent]] * 3, dtype=torch.float64)

    @property
    def sample_bounds(self):
        return torch.stack([self.box_min + 0.1, self.box_max - 0.1], dim=1)  ## the points' bounding box +- 0.1

    def model(self):
        return GeneralNet(ks=[3, 32, 32, 32, 1], act=torch.sin)

    def pretrain_target(self, x):
        ## SDF of the unit ball
        return x.norm(dim=1) - 1

    def res_terms(self, res_lib, opt):
        return {
            "data":    ResidualTerm(res_lib._data,    1.0,                       self.pts, vals=0),
            "eikonal": ResidualTerm(res_lib._eikonal, self.eikonal_weights[opt], self.uniform_in_box(self.n_eikonal)),
            "normal":  ResidualTerm(res_lib._normal,  1.0,                       self.pts, self.normals),
        }

    def update(self, res_terms, model, it, resample=True):
        pass  ## no moving points

    @torch.no_grad()
    def evaluate(self, model, res_lib, final=False):
        ## Model surface samples, refined onto {f=0}
        pts_model = sample_model_surface_binsearch(model, torch.zeros([0, 3], dtype=torch.float64), self.sample_bounds)
        pts_model, ok = project_to_model(model, res_lib, pts_model)
        pts_model = pts_model[ok]
        ## Truth -> model: samples of the mesh projected onto the model surface
        pts_proj, ok_true = project_to_model(model, res_lib, self.pts_surface_true)
        d_t2m = (pts_proj - self.pts_surface_true).norm(dim=1)[ok_true]
        ## Largest connected component of the model surface (marching cubes in the sampling box)
        bounds = self.sample_bounds
        verts, faces = get_mesh(lambda x: model(x.double()), N=128, device=bounds.device, bbox_min=bounds[:, 0], bbox_max=bounds[:, 1])
        if len(pts_model) == 0 or len(faces) == 0:  ## the surface left the sampling box
            return {k: np.nan for k in self.table_keys if k != "loss"}
        _, on_main, n_components, removed_area = largest_component(verts, faces)
        on_main = torch.tensor(on_main(pts_model.cpu().numpy()), device=pts_model.device)
        ## Model -> truth (exact closest points on the mesh) and normal consistency: on the largest component without the
        ## samples over holes of the scan, and on all samples (_all)
        d_m2t, cos, over_hole = self.model_to_true(model, res_lib, pts_model[on_main])
        d_m2t, cos = d_m2t[~over_hole], cos[~over_hole]
        d_m2t_all, cos_all, _ = self.model_to_true(model, res_lib, pts_model)
        ## Training points (point-to-surface error) to the model surface
        d_pts = self.distance_to_model(model, res_lib, self.pts)
        g_eik = res_lib._grad_x_f_batch(model.params, self.pts_eval_eikonal).squeeze(1)

        metrics = {
            **chamfer(d_t2m, d_m2t),
            **{k + "_all": v for k, v in chamfer(d_t2m, d_m2t_all).items() if k != "cd_t2m"},
            "n_components": n_components,
            "removed_area_fraction": removed_area,
            "over_hole_fraction": over_hole.double().mean().item() if len(over_hole) else np.nan,
            "t2m_converged": ok_true.double().mean().item(),
            "normal_consistency": cos.mean().item() if len(cos) else np.nan,
            "normal_consistency_all": cos_all.mean().item() if len(cos_all) else np.nan,
            "interface_rms": rms(d_pts),
            "interface_max": maximum(d_pts),
            "eikonal_rms": rms(g_eik.norm(dim=1) - 1),
        }
        if len(self.pts_held_out):
            metrics["held_out_rms"] = rms(self.distance_to_model(model, res_lib, self.pts_held_out))
        return metrics

    def model_to_true(self, model, res_lib, pts, rim_tol=1e-7):
        """
        Exact distance of model surface samples to the mesh, the cosine between model and mesh normals there, and whether
        the closest point lies on an open boundary edge of the mesh, i.e. the sample is over a hole of the scan.
        """
        empty = torch.zeros(0, dtype=torch.float64)
        if len(pts) == 0:
            return empty, empty, empty.bool()
        closest, d, face = trimesh.proximity.closest_point(self.mesh, pts.cpu().numpy())
        g = res_lib._grad_x_f_batch(model.params, pts).squeeze(1)
        n_true = torch.tensor(self.mesh.face_normals[face], dtype=torch.float64)
        over_hole = np.zeros(len(pts), dtype=bool)
        if len(self.rim_a):
            ab = self.rim_b - self.rim_a
            t = np.clip(((closest[:, None] - self.rim_a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0, 1)
            over_hole = np.linalg.norm(closest[:, None] - (self.rim_a[None] + t[..., None] * ab[None]), axis=-1).min(1) < rim_tol
        to_torch = lambda a, dtype: torch.tensor(a, dtype=dtype, device=pts.device)
        return to_torch(d, torch.float64), (g / g.norm(dim=1, keepdim=True) * n_true).sum(1), to_torch(over_hole, torch.bool)

    @staticmethod
    def distance_to_model(model, res_lib, x):
        pts_proj, ok = project_to_model(model, res_lib, x)
        return (pts_proj - x).norm(dim=1)[ok]


class RockerArm(ImplicitShapeTask):
    mesh_file = "rocker-arm.off"
    n_eikonal = 7000
    paper_data = "notebooks/Data Rocker Arm/{}_over_time_rockerarm_{}.pth"
    heatmap = dict(elevation=25, azimuth=0, gap=0.05, rotate=[("y", 90), ("x", -90), ("y", 225)])  ## Figure 4a


class Bunny(ImplicitShapeTask):
    """Settings of commit 477423d, which produced the stored GN curve of Figure 5: 5000 eikonal points and GaussNewton's line search."""
    mesh_file = "bun_zipper.ply"
    n_eikonal = 5000
    gn_class = GaussNewton
    paper_data = "notebooks/Data Bunny/{}_over_time_bunny_{}.pth"
    heatmap = dict(elevation=10, azimuth=0, gap=0.05, rotate=[("x", 90)])  ## Figure 5a: the mesh has y up
