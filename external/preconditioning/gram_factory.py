import torch
# Calculate Gauss-Newton Gram Matrices

def calculate_A_interface(model, pts):
    dr_dict = model.grad_theta_f(model.params, pts)
    dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
    A_interface = torch.einsum('bi,bj->ij', dr, dr) / len(pts)
    return A_interface.detach()

def calculate_A_eikonal(model, pts):
    dr_dict = model.grad_theta_r_eikonal(model.params, pts)
    dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
    A_eikonal = torch.einsum('bi,bj->ij', dr, dr) / len(pts)
    return A_eikonal

def calculate_A_mean_curvature(model, pts):
    dr_dict = model.grad_theta_r_mean_curvature(model.params, pts)
    dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
    A_mean_curvature = torch.einsum('bi,bj->ij', dr, dr) / len(pts)
    return A_mean_curvature
    

def compute_gram_matrix(model, config):
    pts_boundary = config.get("pts_boundary")
    pts_eikonal = config.get("pts_eikonal")
    pts_surface = config.get("pts_surface")
    loss_weights = config.get("loss_weights")

    A = loss_weights["interface"] * calculate_A_interface(model, pts_boundary)
    
    if loss_weights.get("eikonal", 0) != 0:
        A += loss_weights["eikonal"] * calculate_A_eikonal(model, pts_eikonal)
    
    if loss_weights.get("curvature", 0) != 0:
        A += loss_weights["curvature"] * calculate_A_mean_curvature(model, pts_surface)

    return A.detach()