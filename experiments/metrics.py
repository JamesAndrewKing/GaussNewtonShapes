"""
Geometric error metrics shared by all tasks.
Task-specific parts (exact distance to the true surface, true surface samples) live in the tasks.
"""

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from torch.func import vmap


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
    }


def curvature_stats(residual, params, pts, prefix):
    """RMS and max of a curvature residual (e.g. res_lib._mean_curvature) on surface points."""
    k = vmap(residual, in_dims=(None, 0))(params, pts).reshape(-1).abs()
    return {f"{prefix}_rms": rms(k), f"{prefix}_max": maximum(k)}


def largest_component(verts, faces):
    """
    Largest connected component (by area) of a marching-cubes mesh.
    Returns the component as a trimesh, a function telling which points lie on it (nearest vertex), the number of
    components and the area fraction of the discarded ones.
    """
    mesh = trimesh.Trimesh(verts, faces)
    mesh.merge_vertices()  ## join the duplicate vertices at the seams of chunked marching cubes
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    areas = np.bincount(labels, weights=mesh.area_faces)
    main = areas.argmax()
    vertex_on_main = np.zeros(len(mesh.vertices), dtype=bool)
    vertex_on_main[mesh.faces[labels == main].ravel()] = True
    tree = cKDTree(mesh.vertices)
    on_main = lambda pts: vertex_on_main[tree.query(pts)[1]]
    component = mesh.submesh([np.flatnonzero(labels == main)], append=True)
    return component, on_main, len(areas), 1 - areas[main] / areas.sum()

