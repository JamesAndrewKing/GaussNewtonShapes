import torch
from GINN.numerical_boundary import find_boundary_points_numerically_with_binsearch
from models.net_w_partials import NetWithPartials
from util.sample_utils import precompute_sample_grid

def sample_model_surface_newton(model, init_points, n_iter=10, newton_clip=0.15, tol=1e-8):
    for _ in range(n_iter):
        init_points.requires_grad_(True)
        f_val = model(init_points)
        # Compute gradient
        grad_f = model.grad_x_f(model.params, init_points).squeeze(1)
        # Avoid division by small gradients
        grad_norm_sq = torch.sum(grad_f**2, dim=1, keepdim=True)
        valid = grad_norm_sq.squeeze() > 1e-10
        # Newton step: p_new = p_old - f(p) * grad_f / ||grad_f||^2
        step = torch.zeros_like(init_points)
        step[valid] = (f_val[valid] / grad_norm_sq[valid]) * grad_f[valid]
        step = torch.clamp(step, -newton_clip, newton_clip)  # Clip updates
        init_points = init_points - step.detach()
        # Stop early if all points are close to zero level set
        if torch.max(torch.abs(f_val)) < tol:
            break
    # Filter valid points (close to zero level set)
    valid_mask = torch.abs(f_val) < tol
    valid_mask = valid_mask.squeeze(1)
    points_on_surface = init_points[valid_mask].detach()
    return points_on_surface

def sample_model_surface_binsearch(model, pts_boundary, bound_limit=1.0):
    netp = NetWithPartials.create_from_model(model=model, nz=0, nx=3)
    # bounds = torch.tensor([[-2,2], [-2,2], [-2,2]], dtype=torch.float64)
    bounds = torch.tensor([[-bound_limit,bound_limit], [-bound_limit,bound_limit], [-bound_limit,bound_limit]], dtype=torch.float64)
    grid_find_surface, grid_dist_find_surface, init_grid_resolution = precompute_sample_grid(
            10000, bounds, equidistant=True)
    success, (p_surface, y_sel) = find_boundary_points_numerically_with_binsearch(
        netp=netp, 
        z=torch.zeros([1,0], dtype=torch.float64),
        n_steps=10,
        x_grid=grid_find_surface, 
        x_grid_dist=grid_dist_find_surface, 
        level_set=0.0,
        nf_is_density=True,
        resolution=init_grid_resolution
        )
    if success:
        points_on_surface = p_surface.data
        points_on_surface = torch.cat((pts_boundary, points_on_surface))
    else:
        points_on_surface = pts_boundary
    return points_on_surface