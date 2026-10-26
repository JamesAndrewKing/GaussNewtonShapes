"""
Shared setup of the minimal and developable surface problems (Sections 4.1 and 4.2): an interface loss on a
prescribed boundary curve plus a curvature loss on the moving surface, with an eikonal term for the first eikonal_off_time (50 s) of training time.
Subclasses define the boundary, the true surface (exact distance, region of interest, area) and the figure settings.
"""

import torch
from models.model_architecture import GeneralNet
from training.residuals import ResidualTerm
from util.surface_sampling import sample_model_surface_binsearch, sample_model_surface_newton
from util.error_metrics import chamfer_div
from experiments.metrics import project_to_model, chamfer, curvature_stats, clipped_mesh_area, rms, maximum


class CurvatureTask:
    curvature = "mean_curvature"  ## residual in training.residuals.ResidualLibrary, without the leading "_"
    ## Per optimizer: GN and Adam as for the paper figures; LBFGS chosen by an eikonal sweep (lower weights let it collapse f -> 0)
    eikonal_weights = {"gn": 1e-3, "adam": 1e-3, "lbfgs": 1e-3}
    eikonal_off_time = 50.0       ## seconds of training time after which the eikonal weight is 0, the same for all optimizers
    eikonal_with_boundary = True  ## also enforce the eikonal equation on the boundary points
    refine_surface = False        ## Newton-refine the binary-search surface samples during training
    opt_defaults = {
        "gn":    dict(lr=1e-1, regularization=1e-6, do_line_search=False),
        "adam":  dict(lr=1e-3),
        "lbfgs": dict(line_search_fn="strong_wolfe"),  ## otherwise torch defaults (lr=1, max_iter=20, history_size=100)
    }
    figure_opts = ["adam", "lbfgs", "gn"]  ## optimizers shown in the paper figures, left to right
    table_keys = ["loss", "cd", "cd_t2m", "cd_m2t", "hausdorff", "interface_rms", "curv_rms", "curv_max", "area_err_discrete"]
    pretrain_extent = 1.0  ## pretraining points in [-extent, extent]^3
    sample_extent = 1.0    ## surface samples (training and evaluation) in [-extent, extent]^3

    @staticmethod
    def box(extent):
        return torch.tensor([[-extent, extent]] * 3, dtype=torch.float64)

    @property
    def pretrain_bounds(self):
        return self.box(self.pretrain_extent)

    @property
    def sample_bounds(self):
        return self.box(self.sample_extent)

    @property
    def loss_keys(self):
        return ["data", self.curvature]  ## terms of the reported loss

    def model(self):
        return GeneralNet(ks=[3, 32, 32, 1])

    def res_terms(self, res_lib, opt):
        self.res_lib = res_lib
        pts_eikonal = 2 * torch.rand([1000, 3], dtype=torch.float64) - 1
        if self.eikonal_with_boundary:
            pts_eikonal = torch.cat((pts_eikonal, self.pts_boundary))
        return {
            "data":          ResidualTerm(res_lib._data,                          1.0, self.pts_boundary, vals=0),
            "eikonal":       ResidualTerm(res_lib._eikonal, self.eikonal_weights[opt],  pts_eikonal,       ),
            self.curvature:  ResidualTerm(getattr(res_lib, "_" + self.curvature), 1.0, self.pts_boundary, ),
        }

    def update(self, res_terms, model, t, resample=True):
        """Called before every optimizer step (t: training time in s): loss weight schedule and surface resampling."""
        if t >= self.eikonal_off_time:
            res_terms["eikonal"].weight = 0.0
        if not resample:
            return
        pts_surface = sample_model_surface_binsearch(model, self.pts_boundary, self.sample_bounds)
        if self.refine_surface:
            pts_surface = sample_model_surface_newton(model, self.res_lib, pts_surface)
        res_terms[self.curvature].points = pts_surface

    def mesh_area(self, model, z_range):
        """Mesh area of the model between two z-planes, relative to the analytic area and to the true surface meshed the same way."""
        bbox_min, bbox_max = (torch.tensor(b, dtype=torch.float64) for b in self.mesh_bbox)
        area = lambda f: clipped_mesh_area(f, bbox_min, bbox_max, z_range)
        if getattr(self, "_area_true_mesh", None) is None:
            self._area_true_mesh = area(self.level_set)
        a = area(lambda x: model(x.double()))
        return {"area": a, "area_err_analytic": a / self.area_true - 1, "area_err_discrete": a / self._area_true_mesh - 1}

    @torch.no_grad()
    def evaluate(self, model, res_lib, final=False):
        self.res_lib = res_lib
        params = model.params
        residual = getattr(res_lib, "_" + self.curvature)
        ## Model surface samples, refined onto {f=0}
        pts_model = sample_model_surface_binsearch(model, torch.zeros([0, 3], dtype=torch.float64), self.sample_bounds)
        pts_model, ok = project_to_model(model, res_lib, pts_model)
        pts_model = pts_model[ok]
        pts_region = pts_model[self.in_region(pts_model)]
        ## True surface points projected onto the model
        pts_proj, ok_true = project_to_model(model, res_lib, self.pts_surface_true)
        d_t2m = (pts_proj - self.pts_surface_true).norm(dim=1)[ok_true]
        d_m2t = self.distance_to_true(pts_region)
        ## Interface: distance of the boundary curve to the model surface
        pts_proj, ok_bd = project_to_model(model, res_lib, self.pts_boundary)
        d_bd = (pts_proj - self.pts_boundary).norm(dim=1)[ok_bd]

        metrics = {
            **chamfer(d_t2m, d_m2t),
            "t2m_converged": ok_true.double().mean().item(),
            "cd1_paper": chamfer_div(model, res_lib, self.pts_surface_true.clone()),
            "interface_rms": rms(d_bd),
            "interface_max": maximum(d_bd),
            **curvature_stats(residual, params, pts_region, "curv"),
            **curvature_stats(residual, params, pts_model, "curv_box"),
        }
        if final:
            metrics.update(self.area(model))
        return metrics
