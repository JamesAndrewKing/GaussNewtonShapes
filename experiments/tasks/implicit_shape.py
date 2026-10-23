"""
Implicit neural shapes (notebooks/rockerarm.ipynb, notebooks/bunny.ipynb, Section 4.3): fit f = 0 to the vertices
of a mesh with their normals, regularized by an eikonal term. Settings as for the paper figures (commits cb574c0/5ee82bc),
except the normal residual: the figures used <grad f / |grad f|, n> - 1, the harness uses the unnormalized one of the library.
"""

import numpy as np
import torch
import trimesh
from models.model_architecture import GeneralNet
from training.optimizers import GaussNewtonNew
from training.residuals import ResidualTerm
from util.surface_sampling import sample_model_surface_binsearch
from experiments.metrics import project_to_model, chamfer, rms, maximum

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
    table_keys = ["loss", "cd", "cd_t2m", "cd_m2t", "hausdorff", "interface_rms", "normal_consistency", "eikonal_rms", "held_out_rms"]
    pretrain_extent = 1.5

    def __init__(self, subsample_fraction=1.0):
        mesh = trimesh.load(NOTEBOOKS_DIR + self.mesh_file)
        ## Normalize as in the notebooks: centered, max vertex norm 1
        center = mesh.vertices.mean(axis=0)
        scale = np.linalg.norm(mesh.vertices - center, axis=1).max()
        mesh.apply_translation(-center)
        mesh.apply_scale(1 / scale)
        self.mesh = mesh
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
        ## Model -> truth: exact closest points on the mesh
        closest, d_m2t, face = trimesh.proximity.closest_point(self.mesh, pts_model.cpu().numpy())
        d_m2t = torch.tensor(d_m2t, dtype=torch.float64)
        ## Normal consistency: model normal vs mesh face normal at the closest point
        g = res_lib._grad_x_f_batch(model.params, pts_model).squeeze(1)
        n_true = torch.tensor(self.mesh.face_normals[face], dtype=torch.float64)
        cos = (g / g.norm(dim=1, keepdim=True) * n_true).sum(1)
        ## Training points (point-to-surface error) to the model surface
        d_pts = self.distance_to_model(model, res_lib, self.pts)
        g_eik = res_lib._grad_x_f_batch(model.params, self.pts_eval_eikonal).squeeze(1)

        metrics = {
            **chamfer(d_t2m, d_m2t),
            "t2m_converged": ok_true.double().mean().item(),
            "normal_consistency": cos.mean().item() if len(cos) else np.nan,
            "interface_rms": rms(d_pts),
            "interface_max": maximum(d_pts),
            "eikonal_rms": rms(g_eik.norm(dim=1) - 1),
        }
        if len(self.pts_held_out):
            metrics["held_out_rms"] = rms(self.distance_to_model(model, res_lib, self.pts_held_out))
        return metrics

    @staticmethod
    def distance_to_model(model, res_lib, x):
        pts_proj, ok = project_to_model(model, res_lib, x)
        return (pts_proj - x).norm(dim=1)[ok]


class RockerArm(ImplicitShapeTask):
    mesh_file = "rocker-arm.off"
    n_eikonal = 7000
    paper_data = "notebooks/Data Rocker Arm/{}_over_time_rockerarm_{}.pth"
    heatmap = dict(elevation=30, azimuth=0, gap=0.05)  ## Figure 4a


class Bunny(ImplicitShapeTask):
    mesh_file = "bun_zipper.ply"
    n_eikonal = 4000
    paper_data = "notebooks/Data Bunny/{}_over_time_bunny_{}.pth"
    heatmap = dict(elevation=10, azimuth=0, gap=0.05, up="y")  ## Figure 5a
