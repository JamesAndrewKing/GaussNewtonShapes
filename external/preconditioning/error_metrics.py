import torch
from .surface_sampling import sample_model_surface_newton

def compute_distance(model, true_implicit, pts_surface_model, pts_surface_true, max_dist):
    # Refine model surface samples for exact error calculation
    pts_surface_model = sample_model_surface_newton(model, pts_surface_model)
    # Cut away points further away than the boundary points
    # distances = torch.norm(pts_surface_model[:, :2], dim=1)    
    distances = torch.norm(pts_surface_model, dim=1)    
    pts_surface_model = pts_surface_model[distances <= max_dist]
    # Compute d_2
    with torch.no_grad():
        d_model = model(pts_surface_true).squeeze().square().mean()
        d_true = true_implicit(pts_surface_model).squeeze().square().mean()
        distance = torch.sqrt(d_model + d_true).item()
    return distance

def chamfer_div(model, pts_surface_true):    
    # Flow true points onto the surface
    pts_surface_model = sample_model_surface_newton(model, pts_surface_true)    
    dists = torch.cdist(pts_surface_model, pts_surface_true, p=2) ** 2
    min_dists, _ = dists.min(dim=1)
    distance = min_dists.mean()
    return torch.sqrt(distance).item()