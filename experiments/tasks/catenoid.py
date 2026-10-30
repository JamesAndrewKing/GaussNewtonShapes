"""
Plateau's problem for the catenoid (notebooks/catenoid.ipynb, Section 4.1).
Two coaxial rings of radius c*cosh(h/c) at z=±h.

Goldschmidt sweep (h_over_R given): rings of radius R = cosh(1) at z = ±h with h = h_over_R * R, started from the
cylinder of radius R or from the planes z = ±h (init). A catenoid through the rings exists only for
h/R < H_OVER_R_MAX; the true surface is its stable branch, and otherwise the Goldschmidt solution (the two discs).
The Chamfer distance to the discs is always reported as cd_planes, so each result can be classified.
"""

import math
import torch
from experiments.tasks.curvature_task import CurvatureTask
from util.surface_sampling import sample_model_surface_binsearch
from experiments.metrics import project_to_model, chamfer

A_MAX = 1.19967864  ## minimizer of cosh(a)/a, a = h/c
H_OVER_R_MAX = A_MAX / math.cosh(A_MAX)  ## ~0.6627: no catenoid through the rings beyond this separation


def catenoid_c(h, R, stable=True):
    """c with c*cosh(h/c) = R on the stable (a = h/c < A_MAX) or unstable branch, by bisection; None if none exists."""
    if h / R >= H_OVER_R_MAX:
        return None
    lo, hi = (1e-6, A_MAX) if stable else (A_MAX, 50.0)
    g = lambda a: math.cosh(a) / a - R / h  ## decreasing on the stable branch, increasing on the unstable one
    for _ in range(200):
        mid = (lo + hi) / 2
        if (g(mid) > 0) == stable:
            lo = mid
        else:
            hi = mid
    return h / lo


def catenoid_area(c, h):
    return math.pi * c * (2*h + c*math.sinh(2*h/c))


class Catenoid(CurvatureTask):
    paper_data = "notebooks/Data Catenoid/{}_over_time_catenoid_{}.pth"  ## stored paper runs: metric, optimizer
    ## Mean curvature is minimized on a box larger than the rings, so the surface stays smooth beyond them
    sample_extent = 2.0
    mesh_bbox = ((-2, -2, -1.25), (2, 2, 1.25))  ## for the area and the heatmap
    heatmap = dict(color_range=(0, 0.002), elevation=20, azimuth=0, gap=0.05)  ## Figure 1a

    def __init__(self, c=1.0, h=1.0, init_radius=1.0, h_over_R=None, init="cylinder"):
        self.init = init
        if h_over_R is None:
            self.R = c * math.cosh(h / c)
        else:
            self.R = math.cosh(1.0)
            h = h_over_R * self.R
            c = catenoid_c(h, self.R)
            init_radius = self.R
            self.pretrain_extent = 2.0  ## the sampling box, which contains the rings and both initial shapes
            self.mesh_bbox = ((-2, -2, -min(h + 0.25, 2)), (2, 2, min(h + 0.25, 2)))
        self.c, self.h, self.init_radius = c, h, init_radius
        ## Boundary rings
        phi = torch.linspace(-torch.pi, torch.pi, 1000, dtype=torch.float64)
        ring = lambda z: torch.vstack([self.R * torch.cos(phi), self.R * torch.sin(phi), torch.full_like(phi, z)]).T
        self.pts_boundary = torch.cat([ring(h), ring(-h)])
        ## Discs bounded by the rings (Goldschmidt solution), on a polar grid
        rho, phi_d = torch.meshgrid(torch.linspace(0, 1, 32, dtype=torch.float64) ** 0.5 * self.R,
                                    torch.linspace(-torch.pi, torch.pi, 128, dtype=torch.float64), indexing='ij')
        disc = lambda z: torch.vstack([(rho * torch.cos(phi_d)).flatten(), (rho * torch.sin(phi_d)).flatten(),
                                       torch.full_like(rho.flatten(), z)]).T
        self.pts_discs = torch.cat([disc(h), disc(-h)])
        if c is None:
            ## No catenoid: the true surface is the Goldschmidt solution
            self.pts_surface_true = self.pts_discs
            self.area_true = 2 * math.pi * self.R**2
            self.distance_to_true, self.in_region = self.distance_to_discs, self.near_discs
            return
        ## True surface between the rings (64 x 128 grid in z and phi)
        z, phi = torch.meshgrid(
            torch.linspace(-h, h, 64, dtype=torch.float64),
            torch.linspace(-torch.pi, torch.pi, 128, dtype=torch.float64),
            indexing='ij',
        )
        self.pts_surface_true = self.surface_polar(z.flatten(), phi.flatten())
        self.area_true = catenoid_area(c, h)

    def surface_polar(self, z, phi):
        r = self.c * torch.cosh(z / self.c)
        return torch.vstack([r * torch.cos(phi), r * torch.sin(phi), z]).T

    def level_set(self, p):
        return p[:, 0]**2 + p[:, 1]**2 - (self.c * torch.cosh(p[:, 2] / self.c))**2

    def pretrain_target(self, x):
        if self.init == "planes":
            return x[:, 2]**2 - self.h**2  ## planes z = ±h
        ## Cylinder of radius init_radius
        return x[:, 0]**2 + x[:, 1]**2 - self.init_radius**2

    def near_discs(self, x, tol=0.05):
        return (x[:, :2].norm(dim=1) <= self.R) & (x[:, 2].abs() <= self.h + tol)

    def distance_to_discs(self, x):
        return torch.sqrt((x[:, :2].norm(dim=1) - self.R).clamp_min(0)**2 + (x[:, 2].abs() - self.h)**2)

    def in_region(self, x):
        return x[:, 2].abs() <= self.h

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

    @torch.no_grad()
    def evaluate(self, model, res_lib, final=False):
        metrics = super().evaluate(model, res_lib, final)
        if self.c is not None:
            ## Chamfer distance to the Goldschmidt solution, inside the cylinder of the rings
            pts_model = sample_model_surface_binsearch(model, torch.zeros([0, 3], dtype=torch.float64), self.sample_bounds)
            pts_model, ok = project_to_model(model, res_lib, pts_model)
            pts_model = pts_model[ok]
            pts_proj, ok_true = project_to_model(model, res_lib, self.pts_discs)
            d_t2m = (pts_proj - self.pts_discs).norm(dim=1)[ok_true]
            d_m2t = self.distance_to_discs(pts_model[self.near_discs(pts_model)])
            metrics.update({k + "_planes": v for k, v in chamfer(d_t2m, d_m2t).items()})
        else:
            metrics.update({k + "_planes": metrics[k] for k in ["cd_t2m", "cd_m2t", "cd", "hausdorff"]})  ## same surface
        return metrics

    def area(self, model):
        if self.c is None:
            return {}  ## the discrete reference area needs the catenoid's level set
        return self.mesh_area(model, (-self.h, self.h))
