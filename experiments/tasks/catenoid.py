"""
Plateau's problem for the catenoid (notebooks/catenoid.ipynb, Section 4.1).
Two coaxial rings of radius c*cosh(h/c) at z=±h; the loss is interface + mean curvature (+ eikonal for the first 500 iterations).
"""

import math
import torch
from models.model_architecture import GeneralNet
from training.residuals import ResidualTerm
from util.surface_sampling import sample_model_surface_binsearch
from util.error_metrics import chamfer_div
from experiments.metrics import project_to_model, chamfer, curvature_stats, clipped_mesh_area, rms


class Catenoid:
    loss_keys = ["data", "mean_curvature"]  ## terms of the reported loss
    paper_data = "notebooks/Data Catenoid/{}_over_time_catenoid_{}.pth"  ## stored paper runs: metric, optimizer
    heatmap = dict(  ## curvature heatmap of the best surfaces (Figure 1a)
        residual="_mean_curvature", color_range=(0, 0.002), elevation=20,
        bbox_min=(-2, -2, -1.25), bbox_max=(2, 2, 1.25),
    )
    opt_defaults = {
        "gn":    dict(lr=1e-1, regularization=1e-6, do_line_search=False),
        "adam":  dict(lr=1e-3),
        "lbfgs": dict(),  ## torch defaults (lr=1, max_iter=20, history_size=100), as for the paper figure
    }

    def __init__(self, c=1.0, h=1.0, init_radius=1.0):
        self.c, self.h, self.init_radius = c, h, init_radius
        self.pretrain_bounds = torch.tensor([[-1, 1], [-1, 1], [-1, 1]], dtype=torch.float64)
        ## Mean curvature is minimized on a box larger than the rings, so the surface stays smooth beyond them
        self.sample_bounds = torch.tensor([[-2, 2], [-2, 2], [-2, 2]], dtype=torch.float64)

        ## Boundary rings
        phi = torch.linspace(-torch.pi, torch.pi, 1000, dtype=torch.float64)
        self.pts_boundary = torch.cat([self.surface_polar(torch.full_like(phi, z), phi) for z in [h, -h]])
        ## True surface between the rings (64 x 128 grid in z and phi)
        z, phi = torch.meshgrid(
            torch.linspace(-h, h, 64, dtype=torch.float64),
            torch.linspace(-torch.pi, torch.pi, 128, dtype=torch.float64),
            indexing='ij',
        )
        self.pts_surface_true = self.surface_polar(z.flatten(), phi.flatten())
        self.area_true = math.pi * c * (2*h + c*math.sinh(2*h/c))
        self._area_true_mesh = None

    def surface_polar(self, z, phi):
        r = self.c * torch.cosh(z / self.c)
        return torch.vstack([r * torch.cos(phi), r * torch.sin(phi), z]).T

    def level_set(self, p):
        return p[:, 0]**2 + p[:, 1]**2 - (self.c * torch.cosh(p[:, 2] / self.c))**2

    def distance_to_true(self, x, n_iter=30):
        """
        Exact distance to the catenoid patch |z| <= h.
        The catenoid is a surface of revolution, so this is the distance from (rho, z) to the profile
        curve s -> (c*cosh(s/c), s), minimized over s in [-h, h] with Newton's method.
        """
        c, h = self.c, self.h
        rho, z = x[:, :2].norm(dim=1), x[:, 2]
        s = z.clamp(-h, h)
        for _ in range(n_iter):
            r, dr, ddr = c * torch.cosh(s/c), torch.sinh(s/c), torch.cosh(s/c) / c
            grad = -(rho - r) * dr - (z - s)
            hess = dr**2 - (rho - r) * ddr + 1
            s = (s - grad / hess.clamp_min(1e-6)).clamp(-h, h)
        return torch.sqrt((rho - c * torch.cosh(s/c))**2 + (z - s)**2)

    def model(self):
        return GeneralNet(ks=[3, 32, 32, 1])

    def pretrain_target(self, x):
        ## Cylinder of radius init_radius
        return x[:, 0]**2 + x[:, 1]**2 - self.init_radius**2

    def res_terms(self, res_lib):
        pts_eikonal = 2 * torch.rand([1000, 3], dtype=torch.float64) - 1
        pts_eikonal = torch.cat((pts_eikonal, self.pts_boundary))
        return {
            "data":           ResidualTerm(res_lib._data,           1.0,  self.pts_boundary, vals=0),
            "eikonal":        ResidualTerm(res_lib._eikonal,        1e-3, pts_eikonal,       ),
            "mean_curvature": ResidualTerm(res_lib._mean_curvature, 1.0,  self.pts_boundary, ),
        }

    def update(self, res_terms, model, it):
        """Called before every optimizer step: loss weight schedule and surface resampling."""
        if it == 500:
            res_terms["eikonal"].weight = 0.0
        res_terms["mean_curvature"].points = sample_model_surface_binsearch(model, self.pts_boundary, self.sample_bounds)

    @torch.no_grad()
    def evaluate(self, model, res_lib, final=False):
        params = model.params
        ## Model surface samples, refined onto {f=0}
        pts_model = sample_model_surface_binsearch(model, torch.zeros([0, 3], dtype=torch.float64), self.sample_bounds)
        pts_model, ok = project_to_model(model, res_lib, pts_model)
        pts_model = pts_model[ok]
        pts_model_rings = pts_model[pts_model[:, 2].abs() <= self.h]
        ## True surface points projected onto the model
        pts_proj, ok_true = project_to_model(model, res_lib, self.pts_surface_true)
        d_t2m = (pts_proj - self.pts_surface_true).norm(dim=1)[ok_true]
        d_m2t = self.distance_to_true(pts_model_rings)
        ## Interface: distance of the rings to the model surface
        pts_proj, ok_bd = project_to_model(model, res_lib, self.pts_boundary)
        d_bd = (pts_proj - self.pts_boundary).norm(dim=1)[ok_bd]

        metrics = {
            **chamfer(d_t2m, d_m2t),
            "t2m_converged": ok_true.double().mean().item(),
            "cd1_paper": chamfer_div(model, res_lib, self.pts_surface_true.clone()),
            "interface_rms": rms(d_bd),
            "interface_max": d_bd.max().item(),
            **curvature_stats(res_lib._mean_curvature, params, pts_model_rings, "H_rings"),
            **curvature_stats(res_lib._mean_curvature, params, pts_model, "H_box"),
        }
        if final:
            bbox_min = torch.tensor([-2, -2, -1.25], dtype=torch.float64)
            bbox_max = torch.tensor([2, 2, 1.25], dtype=torch.float64)
            area = lambda f: clipped_mesh_area(f, bbox_min, bbox_max, z_lim=self.h)
            if self._area_true_mesh is None:
                self._area_true_mesh = area(self.level_set)
            metrics["area"] = area(lambda x: model(x.double()))
            ## Relative to the analytic area, and to the true surface meshed the same way (cancels discretization error)
            metrics["area_err_analytic"] = metrics["area"] / self.area_true - 1
            metrics["area_err_mesh"] = metrics["area"] / self._area_true_mesh - 1
        return metrics
