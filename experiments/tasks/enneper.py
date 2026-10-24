"""
Plateau's problem for Enneper's surface (notebooks/enneper.ipynb, Section 4.1).
The boundary is the image of the circle r = r_max under the parametrization
    X(u, v) = (u - u^3/3 + u v^2,  v - v^3/3 + v u^2,  u^2 - v^2).
For r < 1 the patch is a graph over the xy-plane (the Jacobian of (u, v) -> (x, y) has determinant 1 - r^4),
which gives the region of interest and an exact area element.
"""

import math
import torch
from experiments.tasks.curvature_task import CurvatureTask


def enneper(u, v):
    return torch.stack([u - u**3/3 + u*v**2, v - v**3/3 + v*u**2, u**2 - v**2], dim=-1)


class Enneper(CurvatureTask):
    eikonal_weights = {"gn": 1e-3, "adam": 1e-1, "lbfgs": 1e-1}
    paper_data = "notebooks/Data Enneper/{}_over_time_enneper_{}.pth"
    mesh_bbox = ((-0.85, -0.85, -0.85), (0.85, 0.85, 0.85))  ## for the heatmap
    heatmap = dict(color_range=(0, 0.002), elevation=15, azimuth=35, gap=0.1)  ## Figure 2a

    def __init__(self, r_max=0.9):
        self.r_max = r_max
        phi = torch.linspace(-torch.pi, torch.pi, 1000, dtype=torch.float64)
        self.pts_boundary = enneper(r_max * torch.cos(phi), r_max * torch.sin(phi))
        ## True surface (64 x 128 grid in r and phi)
        r, phi = torch.meshgrid(
            torch.linspace(-r_max, r_max, 64, dtype=torch.float64),
            torch.linspace(-torch.pi, torch.pi, 128, dtype=torch.float64),
            indexing='ij',
        )
        self.pts_surface_true = enneper((r * torch.cos(phi)).flatten(), (r * torch.sin(phi)).flatten())
        self.area_true = 2 * math.pi * (r_max**2/2 + r_max**4/2 + r_max**6/6)  ## int (1+r^2)^2 r dr dphi

    def pretrain_target(self, x):
        ## The xy-plane
        return x[:, 2]

    @staticmethod
    def uv_from_xy(x, y, n_steps=20, n_iter=5):
        """
        Invert (u, v) -> (x, y) of the parametrization on the disc r < 1.
        The map has further preimages with r > 1, so Newton's method is continued from the origin along t*(x, y), t: 0 -> 1.
        """
        u, v = torch.zeros_like(x), torch.zeros_like(y)
        for t in torch.linspace(1 / n_steps, 1, n_steps):
            for _ in range(n_iter):
                p = enneper(u, v)
                a, b, c = 1 - u**2 + v**2, 2*u*v, 1 - v**2 + u**2  ## Jacobian [[a, b], [b, c]]
                det = (a*c - b**2).clamp_min(1e-6)
                dx, dy = p[:, 0] - t*x, p[:, 1] - t*y
                u, v = u - (c*dx - b*dy) / det, v - (a*dy - b*dx) / det
        return u, v

    def in_region(self, x, slab=0.1):
        """
        Above or below the patch in xy and within `slab` of it in z. The slab excludes other sheets of the surface:
        the minimal surface continues beyond the boundary, and for r > 1 it folds back over the patch.
        """
        u, v = self.uv_from_xy(x[:, 0], x[:, 1])
        return (u**2 + v**2 <= self.r_max**2) & ((x[:, 2] - (u**2 - v**2)).abs() <= slab)

    def distance_to_true(self, x, n_iter=20):
        """Exact distance to the patch r <= r_max: Gauss-Newton in (u, v), started from the vertical projection."""
        u, v = self.uv_from_xy(x[:, 0], x[:, 1])
        for _ in range(n_iter):
            res = x - enneper(u, v)
            X_u = torch.stack([1 - u**2 + v**2, 2*u*v, 2*u], dim=-1)
            X_v = torch.stack([2*u*v, 1 - v**2 + u**2, -2*v], dim=-1)
            J = torch.stack([X_u, X_v], dim=-1)  ## [B, 3, 2]
            delta = torch.linalg.solve(J.mT @ J, (J.mT @ res[..., None]))[..., 0]
            u, v = u + delta[:, 0], v + delta[:, 1]
            scale = (self.r_max / (u**2 + v**2).sqrt().clamp_min(1e-12)).clamp(max=1)  ## stay on the patch
            u, v = u * scale, v * scale
        return (x - enneper(u, v)).norm(dim=1)

    def area(self, model, n=600, n_iter=10):
        """
        Area of the model surface over the region, as a quadrature over an xy-grid of
            |grad f| / |df/dz|   (model)   and   (1 + r^2) / (1 - r^2)   (true surface).
        The same grid and mask are used for both, so the boundary discretization cancels in area_err_discrete.
        """
        s = torch.linspace(-1.2, 1.2, n, dtype=torch.float64)
        x, y = (g.flatten() for g in torch.meshgrid(s, s, indexing='ij'))
        u, v = self.uv_from_xy(x, y)
        r2 = u**2 + v**2
        mask = r2 <= self.r_max**2
        x, y, r2 = x[mask], y[mask], r2[mask]
        pts = torch.stack([x, y, (u**2 - v**2)[mask]], dim=1)
        for _ in range(n_iter):  ## Newton in z onto the model surface
            g = self.res_lib._grad_x_f_batch(model.params, pts).squeeze(1)
            pts[:, 2] -= model(pts).squeeze(1) / g[:, 2]
        g = self.res_lib._grad_x_f_batch(model.params, pts).squeeze(1)
        cell = (s[1] - s[0]).item()**2
        a = (g.norm(dim=1) / g[:, 2].abs()).sum().item() * cell
        a_discrete = ((1 + r2) / (1 - r2)).sum().item() * cell
        return {"area": a, "area_err_analytic": a / self.area_true - 1, "area_err_discrete": a / a_discrete - 1}
