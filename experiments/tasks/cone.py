"""
Developable surface: mantle of a truncated cone (notebooks/cone.ipynb, Section 4.2).
Circles of radius 1 at z=0 and radius 0.4 at z=1; the loss is interface + Gauss curvature.
Settings as for the paper figure (commit 3531425): optimizer-specific eikonal weights, no eikonal on the boundary,
Newton-refined surface samples.
"""

import math
import torch
from experiments.tasks.curvature_task import CurvatureTask


class Cone(CurvatureTask):
    curvature = "gauss_curvature"
    eikonal_weights = {"gn": 1e-3, "adam": 1e-1, "lbfgs": 1e-3}
    eikonal_with_boundary = False
    refine_surface = True
    opt_defaults = {**CurvatureTask.opt_defaults, "adam": dict(lr=1e-4)}
    paper_data = "notebooks/Data Cone/{}_over_time_cone_{}.pth"
    mesh_bbox = ((-2, -2, -0.05), (2, 2, 1.05))
    heatmap = dict(color_range=(0, 0.005), elevation=25, azimuth=0, gap=0.08)  ## Figure 3a
    figure_opts = ["adam", "gn"]  ## LBFGS diverges

    def __init__(self, r_bottom=1.0, r_top=0.4, height=1.0):
        self.r_bottom, self.r_top, self.height = r_bottom, r_top, height
        t = torch.linspace(0, 2 * torch.pi, 333, dtype=torch.float64)
        self.pts_boundary = torch.cat([self.circle(r, z, t) for r, z in [(r_bottom, 0.0), (r_top, height)]])
        ## True surface (64 x 128 grid in z and phi)
        z, phi = torch.meshgrid(
            torch.linspace(0, height, 64, dtype=torch.float64),
            torch.linspace(-torch.pi, torch.pi, 128, dtype=torch.float64),
            indexing='ij',
        )
        self.pts_surface_true = self.circle(self.radius(z.flatten()), z.flatten(), phi.flatten())
        self.area_true = math.pi * (r_bottom + r_top) * math.hypot(r_bottom - r_top, height)

    def radius(self, z):
        return self.r_bottom + (self.r_top - self.r_bottom) * z / self.height

    @staticmethod
    def circle(r, z, t):
        return torch.vstack([r * torch.sin(t), r * torch.cos(t), torch.zeros_like(t) + z]).T

    def level_set(self, p):
        return p[:, 0]**2 + p[:, 1]**2 - self.radius(p[:, 2])**2

    def pretrain_target(self, x):
        ## Cylinder of radius 1
        return x[:, 0]**2 + x[:, 1]**2 - 1

    def in_region(self, x):
        return (x[:, 2] >= 0) & (x[:, 2] <= self.height)

    def distance_to_true(self, x):
        """Exact distance to the mantle: distance from (rho, z) to the segment from (r_bottom, 0) to (r_top, height)."""
        p = torch.stack([x[:, :2].norm(dim=1), x[:, 2]], dim=1)
        a = torch.tensor([self.r_bottom, 0.0], dtype=torch.float64)
        ab = torch.tensor([self.r_top - self.r_bottom, self.height], dtype=torch.float64)
        t = (((p - a) @ ab) / ab.dot(ab)).clamp(0, 1)
        return (p - a - t[:, None] * ab).norm(dim=1)

    def area(self, model):
        return self.mesh_area(model, (0.0, self.height))
