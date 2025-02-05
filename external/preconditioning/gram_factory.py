import torch
# Approximate Hessians

def calculate_A_interface(model, pts):
    """
    Calculate the diagonal of boundary integral matrix A_bndry.
    
    Args:
        params: Model parameters as a dictionary.
        pts_bndry: Boundary points as a tensor of shape [n_points, n_dims].
        vphi: Function mapping parameters and points to phi=d_theta f.
        
    Returns:
        torch.Tensor: Gram matrix preconditioner for the boundary loss [n_params, n_params].
    """
    # Compute φ_i at boundary points
    phi_at_pts = model.vphi(model.params, pts)  # Returns a dict

    # Aggregate φ_i into a single tensor
    phi_at_pts_tensor = torch.cat([p.flatten(start_dim=1) for p in phi_at_pts.values()], dim=1)  # Shape: [n_points, n_params]
    A_bndry = torch.einsum('bi,bj->ij', phi_at_pts_tensor, phi_at_pts_tensor) / len(pts)

    return A_bndry.detach()

def calculate_A_eikonal(model, pts):
    dr_dict = model.v_d_theta_f_eikonal(model.params, pts)
    dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
    A_eikonal = torch.einsum('bi,bj->ij', dr, dr) / len(pts)

    return A_eikonal

# def calculate_A_eikonal(model, pts):
#     dr_dict = model.v_d_theta_f_eikonal(model.params, pts)
#     dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
#     grad_phi_at_pts = model.v_phi_x(model.params, pts)
#     grad_phi_tensor = torch.cat([p.flatten(start_dim=1) for p in grad_phi_at_pts.values()], dim=1)
#     norm_grad_f = torch.norm(model.vf_x(model.params, pts), dim=-1, keepdim=True)
#     dr = dr / torch.sqrt(norm_grad_f + torch.ones_like(norm_grad_f)).view(-1, 1)
#      # Reshape to separate the dimensions
#     n_points, n_params_times_dims = grad_phi_tensor.shape
#     n_dims = pts.shape[1]
#     n_params = n_params_times_dims // n_dims
#     grad_phi_tensor = grad_phi_tensor.reshape(n_points, n_params, n_dims)  # Shape: [n_points, n_params, n_dims]
#     grad_phi_tensor = grad_phi_tensor * torch.sqrt(norm_grad_f / (norm_grad_f + torch.ones_like(norm_grad_f)))
    
#     A_eikonal = (torch.einsum('bi,bj->ij', dr, dr) + torch.einsum("bik,bik->i", grad_phi_tensor, grad_phi_tensor)) / len(pts)

#     return A_eikonal

# def calculate_A_eikonal(model, pts):
#     """
#     Calculate the diagonal of boundary integral matrix A_bndry.
    
#     Args:
#         params: Model parameters as a dictionary.
#         pts_bndry: Boundary points as a tensor of shape [n_points, n_dims].
#         vphi_x: Function mapping parameters and points to d_x phi.
#         vf_x: Function mapping parameters and points to d_x f
        
#     Returns:
#         torch.Tensor: Gram matrix preconditioner for the eikonal loss [n_params, n_params].
#     """
#     grad_phi_at_pts = model.v_phi_x(model.params, pts)  # Returns a dict
#     grad_f_at_pts = model.vf_x(model.params, pts)  # Returns a tensor of shape [n_points, 1, n_dims]
#     grad_f_at_pts = grad_f_at_pts / (torch.norm(grad_f_at_pts, dim=-1, keepdim=True)+1e-5)

#     # Aggregate gradients into a single tensor
#     grad_phi_tensor = torch.cat([p.flatten(start_dim=1) for p in grad_phi_at_pts.values()], dim=1)  # Shape: [n_points, n_params * n_dims]
#     # Reshape to separate the dimensions
#     n_points, n_params_times_dims = grad_phi_tensor.shape
#     n_dims = pts.shape[1]
#     n_params = n_params_times_dims // n_dims
#     grad_phi_tensor = grad_phi_tensor.reshape(n_points, n_params, n_dims)  # Shape: [n_points, n_params, n_dims]
#     dot_product_term = torch.einsum("bik,bik->i", grad_phi_tensor, grad_f_at_pts)
#     A_eikonal = torch.einsum('i,j->ij', dot_product_term, dot_product_term)/n_points
    
#     return A_eikonal.detach()


def calculate_A_laplacian(model, pts):
    """
    Calculate the diagonal of boundary integral matrix A_bndry.
    
    Args:
        params: Model parameters as a dictionary.
        pts_bndry: Boundary points as a tensor of shape [n_points, n_dims].
        v_phi_laplace: Function mapping parameters and points to laplacian of phi.
        
    Returns:
        torch.Tensor: Gram matrix preconditioner for the laplacian loss [n_params, n_params].
    """
    # Compute Laplacians of φ_i at points
    lap_phi_at_pts = model.v_phi_laplace(model.params, pts)  # Returns a dict
    
    # Aggregate Laplacians into a single tensor
    lap_phi_tensor = torch.cat([p.flatten(start_dim=1) for p in lap_phi_at_pts.values()], dim=1)  # Shape: [n_points, n_params]
    A_laplacian = torch.einsum('bi,bj->ij', lap_phi_tensor, lap_phi_tensor) / len(pts)

    return A_laplacian.detach()


def calculate_A_mean_curvature(model, pts):
    dr_dict = model.v_d_theta_f_mean_curvature(model.params, pts)
    dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
    A_mean_curvature = torch.einsum('bi,bj->ij', dr, dr) / len(pts)

    return A_mean_curvature
    

def compute_gram_matrix(model, config):
    """
    Compute the diagonal of the Hessian matrix.
    
    Args:
        params: Model parameters as a dictionary.
        pts_boundary: Boundary points tensor of shape [n_points, n_dims].
        pts_corners: Corner points tensor of shape [n_points, n_dims].
        pts_space: Space points tensor of shape [n_points, n_dims].
        loss_weights: Dictionary containing the weights of the different losses.
        
    Returns:
        torch.Tensor: Preconditioning gram matrix [n_params, n_params].
    """
    pts_boundary = config.get("pts_boundary")
    pts_corners = config.get("pts_corners")
    pts_eikonal = config.get("pts_eikonal")
    pts_space = config.get("pts_space")
    pts_surface = config.get("pts_surface")
    loss_weights = config.get("loss_weights")


    # Add terms conditionally based on the presence of keys in the loss_weights dictionary
    A = loss_weights["interface"] * calculate_A_interface(model, pts_boundary)

    if loss_weights.get("interface_corners", 0) != 0:
        A += loss_weights["interface"] * loss_weights["interface_corners"] * calculate_A_interface(model, pts_corners)
    
    if loss_weights.get("eikonal", 0) != 0:
        A += loss_weights["eikonal"] * calculate_A_eikonal(model, pts_eikonal)

    if loss_weights.get("laplacian", 0) != 0:
        A += loss_weights["laplacian"] * calculate_A_laplacian(model, pts_space)
    
    if loss_weights.get("curvature", 0) != 0:
        A += loss_weights["curvature"] * calculate_A_mean_curvature(model, pts_surface)

    return A.detach()