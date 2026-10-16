"""
Geometric error metrics shared by all tasks.
Task-specific parts (exact distance to the true surface, true surface samples) live in the tasks.
"""

import numpy as np
import torch
import trimesh
from torch.func import vmap
from util.visualization.utils_mesh import get_mesh


@torch.no_grad()
def project_to_model(model, res_lib, x, n_iter=20, newton_clip=0.15, tol=1e-8):
    """
    Orthogonal Newton projection of x onto the model surface {f=0} (Appendix B).
    Unlike util.surface_sampling.sample_model_surface_newton, all points are returned (in input order)
    together with a mask of the converged ones, so that distances ||x - P(x)|| can be computed.
    """
    for _ in range(n_iter):
        f = model(x)
        g = res_lib._grad_x_f_batch(model.params, x).squeeze(1)
        step = f * g / g.square().sum(1, keepdim=True).clamp_min(1e-20)
        x = x - step.clamp(-newton_clip, newton_clip)
    converged = model(x).abs().squeeze(1) < tol
    return x, converged


## Both are nan for empty inputs, e.g. if the surface left the region of interest
def rms(d):
    return d.square().mean().sqrt().item()


def maximum(d):
    return d.max().item() if len(d) else np.nan


def chamfer(d_true_to_model, d_model_to_true):
    """Two-sided Chamfer distance from the point-to-surface distances in both directions."""
    t2m, m2t = rms(d_true_to_model), rms(d_model_to_true)
    return {
        "cd_t2m": t2m,
        "cd_m2t": m2t,
        "cd": 0.5 * (t2m + m2t),
        "hausdorff": max(maximum(d_true_to_model), maximum(d_model_to_true)),
    }


def curvature_stats(residual, params, pts, prefix):
    """RMS and max of a curvature residual (e.g. res_lib._mean_curvature) on surface points."""
    k = vmap(residual, in_dims=(None, 0))(params, pts).reshape(-1).abs()
    return {f"{prefix}_rms": rms(k), f"{prefix}_max": maximum(k)}


def clipped_mesh_area(f, bbox_min, bbox_max, z_range, N=256, chunks=2):
    """Area of the marching-cubes mesh of {f=0}, clipped to z_range[0] <= z <= z_range[1]."""
    verts, faces = get_mesh(f, N=N, device=bbox_min.device, bbox_min=bbox_min, bbox_max=bbox_max, chunks=chunks)
    mesh = trimesh.Trimesh(verts, faces)
    mesh = mesh.slice_plane(plane_origin=[0, 0, z_range[0]], plane_normal=[0, 0, 1])
    mesh = mesh.slice_plane(plane_origin=[0, 0, z_range[1]], plane_normal=[0, 0, -1])
    return float(mesh.area) if len(mesh.faces) else np.nan
