"""
Jet engine bracket GINN (notebooks/jeb.ipynb, Section 4.4): interface, interface normals, design region, eikonal,
surface strain and connectedness (persistent homology). Settings of the runs behind Figures 6 and 7 (commit 54cd767):
strain weight 1e-5, without (Figure 6) or with (Figure 7, strain_clip=1000) clipping, iso level 0.1, 4096 surface points,
starting from the pretrained model stored with those runs.
"""

import numpy as np
import torch
import trimesh
import igl
from scipy import ndimage
from torch.func import vmap
from models.wire import ConditionalWIRE
from models.net_w_partials import NetWithPartials
from GINN.ph.ph_manager import PHManager
from GINN.shape_boundary_helper import ShapeBoundaryHelper
from training.optimizers import GaussNewton
from training.residuals import ResidualTerm, ResidualLibrary, compute_loss
from experiments.metrics import project_to_model, rms, maximum

NOTEBOOKS_DIR = __file__.rsplit("experiments", 1)[0] + "notebooks/"
DATA_DIR = NOTEBOOKS_DIR + "Data JEB/"


def subsample_idxs(pts, count, generator=None):
    return torch.randperm(pts.size(0), generator=generator, device="cpu")[:count].to(pts.device)


def subsample(pts, count, generator=None):
    return pts[subsample_idxs(pts, count, generator)]


def distance_filter(pts, pts_filter, th_dist=0.025):
    """Keep the points further than th_dist from all of pts_filter."""
    dist = torch.min(torch.norm(pts[:, None, :] - pts_filter[None, :, :], dim=2), dim=1)[0]
    return pts[dist > th_dist]


class JEB:
    opt_defaults = {
        "gn":    dict(lr=2e-2, regularization=1e-7, do_line_search=True),
        "adam":  dict(lr=3e-4),
        "lbfgs": dict(line_search_fn="strong_wolfe"),
    }
    gn_class = GaussNewton
    figure_opts = ["gn", "adam"]  ## as in Figures 6 and 7
    table_keys = ["loss", "interface_rms", "normal_consistency", "design_region_violation", "design_region_max_depth",
                  "n_components", "strain_mean", "strain_median", "strain_p95", "eikonal_rms"]
    z = 0.05  ## latent code of the conditional WIRE
    heatmap = dict(color_range=(-0.6, 2.3), ticks=[-0.5, 0, 0.5, 1, 1.5, 2], elevation=50, azimuth=45, gap=0.05)  ## Figures 6a, 7a

    def __init__(self, strain_weight=1e-5, strain_clip=None, normal="unnormalized", pretrained="paper"):
        self.strain_weight, self.strain_clip, self.normal, self.pretrained = strain_weight, strain_clip, normal, pretrained
        load = lambda name: torch.from_numpy(np.load(DATA_DIR + name)).to(device=torch.get_default_device(), dtype=torch.float64)
        self.bounds = load('bounds.npy')
        self.mesh_bbox = (tuple(self.bounds[:, 0].tolist()), tuple(self.bounds[:, 1].tolist()))
        self.mesh_env = trimesh.load(DATA_DIR + '411_for_envelope.stl')
        self.pts_interface_all, self.interface_normals_all = load('interface_points.npy'), load('interface_normals.npy')
        self.pts_outside_all, self.pts_inside_all = load('pts_outside.npy'), load('pts_inside.npy')
        self.pts_out_far_all = load('pts_far_outside.npy')
        self.pts_near_interface_all = load('pts_around_interface_outside_env_10mm.npy')
        self.pts_on_env = load('pts_on_env.npy')
        ## Signed distance to the envelope: targets of the design region loss and of the pretraining
        self.design_region_pts_list = [self.pts_out_far_all, self.pts_outside_all, self.pts_near_interface_all, self.pts_on_env]
        self.design_region_sdf_list = [self.envelope_sdf(pts) for pts in self.design_region_pts_list]
        ## Fixed evaluation samples
        gen = torch.Generator().manual_seed(0)
        idxs = subsample_idxs(self.pts_interface_all, 4096, gen)
        self.eval_interface, self.eval_normals = self.pts_interface_all[idxs], self.interface_normals_all[idxs]
        self.eval_outside = torch.vstack([subsample(pts, 4096, gen) for pts in self.design_region_pts_list[:3]])
        self.eval_outside = self.eval_outside[self.envelope_sdf(self.eval_outside) > 0]
        self.eval_eikonal = torch.vstack([subsample(self.pts_inside_all, 2048, gen), subsample(self.pts_outside_all, 2048, gen)])

    def envelope_sdf(self, pts):
        sdf, _, _, _ = igl.signed_distance(pts.cpu().numpy(), self.mesh_env.vertices, self.mesh_env.faces, sign_type=igl.SIGNED_DISTANCE_TYPE_DEFAULT)
        return torch.from_numpy(sdf).to(device=pts.device, dtype=torch.float64)

    def is_inside_envelope(self, pts):
        return (self.envelope_sdf(pts) < 0).cpu()

    def model(self):
        model = ConditionalWIRE([4, 32, 32, 32, 1], False, 18, 1, 6).double()
        model.params = dict(model.named_parameters())
        return model

    def pretrain(self, model, n_iters=3000, batch=8 * 4096):
        """Pretrain on the SDF of the design region (Adam, lr 1e-3), or load the model pretrained for the paper runs."""
        if self.pretrained == "paper":
            model.load_state_dict(torch.load(NOTEBOOKS_DIR + "results/model-Adam-pretrain.pt"))
            return
        pts_sdf = torch.vstack([self.pts_inside_all, self.pts_outside_all, self.pts_out_far_all, self.pts_on_env])
        sdf = self.envelope_sdf(pts_sdf)
        res_terms_pre = {"sdf": ResidualTerm(ResidualLibrary(model)._data, 1, pts_sdf, vals=sdf)}
        optim = torch.optim.Adam(model.parameters(), 1e-3)
        for i in range(n_iters):
            optim.zero_grad()
            idxs = subsample_idxs(pts_sdf, batch)
            res_terms_pre["sdf"].points, res_terms_pre["sdf"].vals = pts_sdf[idxs], sdf[idxs]
            compute_loss(model.params, res_terms_pre).backward()
            optim.step()

    def res_terms(self, res_lib, opt):
        model = res_lib.model
        netp = NetWithPartials.create_from_model(model, nz=1, nx=3)
        self.ph_manager = PHManager(
            mp_pool=None, nx=3, bounds=self.bounds, netp=netp, scc_n_grid_points=64,
            func_inside_envelope=self.is_inside_envelope, ph_1_hole_level=0.1, ph_loss_target_betti=[1, 0, 0],
            iso_level=0.1, ph_loss_sub0_points=False, ph_loss_super0_points=False, simjeb_root_dir="", problem_str="",
            ph_loss_maxdim=0, is_density=False,
        )
        self.shape_boundary_helper = ShapeBoundaryHelper(
            nx=3, bounds=self.bounds, netp=netp, n_points_surface=4096, equidistant_init_grid=64, level_set=0.0,
            do_uniform_resampling=True, surf_pts_lr=0.01, surf_pts_n_iter=10, surf_pts_prec_eps=1.0e-3,
            surf_pts_converged_interval=1, surf_pts_use_newton=True, surf_pts_newton_clip=0.15,
            surf_pts_inflate_bounds_amount=0.05, surf_pts_uniform_n_iter=10, surf_pts_uniform_nof_neighbours=16,
            surf_pts_uniform_stepsize=0.75, surf_pts_uniform_n_iter_reproj=5, surf_pts_uniform_prec_eps=1.0e-3,
            surf_pts_uniform_min_count=1000, surf_pts_surpress_tqdm=True, surf_pts_uniform_reproject_surpress_tqdm=True,
        )
        strain = res_lib._strain
        if self.strain_clip is not None:
            strain = lambda *args: torch.clamp(res_lib._strain(*args), max=self.strain_clip)
        normal = {"unnormalized": res_lib._normal_unnormalized, "normalized": res_lib._normal}[self.normal]
        empty = torch.zeros([0, 3], dtype=torch.float64)
        ## Points are set in update() before every step
        return {
            "interface":     ResidualTerm(res_lib._data,           1e+1,               empty, vals=0),
            "design_region": ResidualTerm(res_lib._design_region,  1e+3,               empty, vals=0.01),
            "eikonal":       ResidualTerm(res_lib._eikonal,        1e+0,               empty),
            "strain":        ResidualTerm(strain,                  self.strain_weight, empty),
            "connectedness": ResidualTerm(res_lib._connectedness,  1e+3,               empty, vals=0),
            "normal":        ResidualTerm(normal,                  1e+1,               empty, vals=None),
        }

    def reported_loss(self, unweighted_losses, res_terms):
        """The weighted total loss, as plotted in Figures 6 and 7."""
        return sum(res_terms[k].weight * l.item() for k, l in unweighted_losses.items())

    def surface_pts(self, model):
        """Surface points for the strain: inside the design region and away from the interfaces."""
        z = self.z + torch.zeros([1, 1], dtype=torch.float64)
        pts = self.shape_boundary_helper.get_surface_pts(z=z)
        pts = torch.zeros([0, 3], dtype=torch.float64) if pts is None else pts.data
        pts = pts[self.is_inside_envelope(pts)]
        return distance_filter(pts, subsample(self.pts_interface_all, 8192), 0.04)

    @torch.no_grad()
    def update(self, res_terms, model, it, resample=True):
        """Called before every optimizer step: new samples for all terms, as in the notebook."""
        if not resample:
            return
        idxs = subsample_idxs(self.pts_interface_all, 4096)
        res_terms["interface"].points = self.pts_interface_all[idxs]
        res_terms["normal"].points, res_terms["normal"].vals = self.pts_interface_all[idxs], self.interface_normals_all[idxs]
        res_terms["eikonal"].points = torch.vstack([subsample(self.pts_inside_all, 2048), subsample(self.pts_outside_all, 2048)])
        ## Design region: sample from four sets like in the GINN paper, then drop the inactive points and correct the weight
        idxs_list = [subsample_idxs(pts, 4096) for pts in self.design_region_pts_list]
        dr = res_terms["design_region"]
        dr.points = torch.vstack([pts[idxs] for pts, idxs in zip(self.design_region_pts_list, idxs_list)])
        dr.vals = torch.hstack([sdf[idxs] for sdf, idxs in zip(self.design_region_sdf_list, idxs_list)])
        n_all = len(dr.points)
        mask = dr.eval(model.params).squeeze().nonzero().squeeze(-1)
        dr.points, dr.vals = dr.points[mask], dr.vals[mask]
        if len(dr.points) > 0:
            dr.weight = 1e+3 / n_all * len(dr.points)
        res_terms["strain"].points = self.surface_pts(model)
        success, x_in = self.ph_manager.calc_ph_loss_cripser(z=self.z + torch.zeros([1, 1], dtype=torch.float64), return_for_GN=True)
        res_terms["connectedness"].points = x_in.to(dtype=torch.float64)

    def surface_attribute(self, model, verts):
        """log10 of the strain residual sqrt(k1^2 + k2^2) on mesh vertices, as plotted in Figures 6a and 7a."""
        res_lib = ResidualLibrary(model)
        k = vmap(res_lib._strain, in_dims=(None, 0))(model.params, torch.tensor(verts, dtype=torch.float64)).squeeze()
        return k.log10().detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def evaluate(self, model, res_lib, final=False):
        params = model.params
        ## Interface: distance of the interface points to the surface, and normal alignment there
        pts_proj, ok = project_to_model(model, res_lib, self.eval_interface)
        d_interface = (pts_proj - self.eval_interface).norm(dim=1)[ok]
        g = res_lib._grad_x_f_batch(params, self.eval_interface).squeeze(1)
        cos = (g / g.norm(dim=1, keepdim=True) * self.eval_normals).sum(1)
        ## Design region: material (f < 0) outside the envelope, and how far outside it reaches
        f_out = model(self.eval_outside).squeeze(1)
        violating = f_out < 0
        depth = self.envelope_sdf(self.eval_outside[violating]) if violating.any() else torch.zeros(1, dtype=torch.float64)
        ## Connectedness: connected components of the solid {f < 0} on a 128^3 grid of the bounding box
        axes = [torch.linspace(lo, hi, 128, dtype=torch.float64) for lo, hi in self.bounds.tolist()]
        grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)
        solid = torch.cat([model(chunk).squeeze(1) < 0 for chunk in grid.split(2**20)]).reshape(128, 128, 128)
        n_components = ndimage.label(solid.cpu().numpy())[1]
        ## Strain k1^2 + k2^2 on the surface points used for the strain loss
        pts = self.surface_pts(model)
        strain = vmap(res_lib._strain, in_dims=(None, 0))(params, pts).reshape(-1).square() if len(pts) else torch.zeros(0)
        q = lambda p: torch.quantile(strain, p).item() if len(strain) else np.nan
        g_eik = res_lib._grad_x_f_batch(params, self.eval_eikonal).squeeze(1)
        return {
            "interface_rms": rms(d_interface),
            "interface_max": maximum(d_interface),
            "interface_converged": ok.double().mean().item(),
            "normal_consistency": cos.mean().item(),
            "design_region_violation": violating.double().mean().item(),
            "design_region_max_depth": depth.max().item(),
            "n_components": n_components,
            "connected": float(n_components == 1),
            "strain_mean": strain.mean().item() if len(strain) else np.nan,
            "strain_median": q(0.5),
            "strain_p95": q(0.95),
            "eikonal_rms": rms(g_eik.norm(dim=1) - 1),
        }
