"""
Plateau's problem for the catenoid (notebooks/catenoid.ipynb, Section 4.1).
Two coaxial rings of radius c*cosh(h/c) at z=±h.
"""

import math
import torch
from experiments.tasks.curvature_task import CurvatureTask


class Catenoid(CurvatureTask):
    paper_data = "notebooks/Data Catenoid/{}_over_time_catenoid_{}.pth"  ## stored paper runs: metric, optimizer
    ## Mean curvature is minimized on a box larger than the rings, so the surface stays smooth beyond them
    sample_extent = 2.0
    mesh_bbox = ((-2, -2, -1.25), (2, 2, 1.25))  ## for the area and the heatmap
    heatmap = dict(color_range=(0, 0.002), elevation=20, azimuth=0, gap=0.05)  ## Figure 1a

    def __init__(self, c=1.0, h=1.0, init_radius=1.0):
        self.c, self.h, self.init_radius = c, h, init_radius
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

    def surface_polar(self, z, phi):
        r = self.c * torch.cosh(z / self.c)
        return torch.vstack([r * torch.cos(phi), r * torch.sin(phi), z]).T

    def level_set(self, p):
        return p[:, 0]**2 + p[:, 1]**2 - (self.c * torch.cosh(p[:, 2] / self.c))**2

    def pretrain_target(self, x):
        ## Cylinder of radius init_radius
        return x[:, 0]**2 + x[:, 1]**2 - self.init_radius**2

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

    def area(self, model):
        return self.mesh_area(model, (-self.h, self.h))
