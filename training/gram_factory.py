import torch
import numpy as np
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



@torch.no_grad()
def compute_jacobian_and_residual(model, config):
    def flatten_grad_dict(grad_dict):
        return torch.cat([p.flatten(start_dim=1) for p in grad_dict.values()], dim=1)

    loss_weights = config["loss_weights"]
    params = model.params

    J_blocks = []
    r_blocks = []

    # Interface
    if "pts_boundary" in config and loss_weights.get("interface", 0.0) > 0:
        pts = config["pts_boundary"]
        N = pts.shape[0]
        r = model.f(params, pts).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["interface"] / N) * r
        )
        dr = flatten_grad_dict(model.grad_theta_f(params, pts))
        J_blocks.append(
            np.sqrt(loss_weights["interface"] / N) * dr
        )

    # Eikonal
    if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
        pts = config["pts_eikonal"]
        N = pts.shape[0]
        r = model.r_eikonal(params, pts).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["eikonal"] / N) * r
        )
        dr = flatten_grad_dict(model.grad_theta_r_eikonal(params, pts))
        J_blocks.append(
            np.sqrt(loss_weights["eikonal"] / N) * dr
        )

    # Curvature
    if "pts_surface" in config and loss_weights.get("curvature", 0.0) > 0:
        pts = config["pts_surface"]
        N = pts.shape[0]
        r = model.r_mean_curvature(params, pts).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["curvature"] / N) * r
        )
        dr = flatten_grad_dict(model.grad_theta_r_mean_curvature(params, pts))
        J_blocks.append(
            np.sqrt(loss_weights["curvature"] / N) * dr
        )

    # Stack full residual and Jacobian
    r = torch.cat(r_blocks, dim=0)
    J = torch.cat(J_blocks, dim=0).T

    return J.detach(), r.detach()


def compute_residual(model, config):
    def flatten_grad_dict(grad_dict):
        return torch.cat([p.flatten(start_dim=1) for p in grad_dict.values()], dim=1)

    loss_weights = config["loss_weights"]
    params = model.params

    r_blocks = []

    # Interface
    if "pts_boundary" in config and loss_weights.get("interface", 0.0) > 0:
        pts = config["pts_boundary"]
        N = pts.shape[0]
        r = model.f(params, pts).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["interface"] / N) * r
        )

    # Eikonal
    if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
        pts = config["pts_eikonal"]
        N = pts.shape[0]
        r = model.r_eikonal(params, pts).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["eikonal"] / N) * r
        )
        
    # Curvature
    if "pts_surface" in config and loss_weights.get("curvature", 0.0) > 0:
        pts = config["pts_surface"]
        N = pts.shape[0]
        r = model.r_mean_curvature(params, pts).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["curvature"] / N) * r
        )

    # Stack full residual and Jacobian
    r = torch.cat(r_blocks, dim=0)

    return r.detach()

def compute_JTJ(model, config):
    def flatten_layer_grads(grad_dict, layer_name):
        # Dynamically extract keys corresponding to the given layer
        layer_keys = [k for k in grad_dict if layer_name in k]  
        return torch.cat([grad_dict[k].flatten(start_dim=1) for k in layer_keys], dim=1)

    loss_weights = config["loss_weights"]
    params = model.params

    N = 0
    if "pts_boundary" in config and loss_weights.get("interface", 0.0) > 0:
        N += config["pts_boundary"].shape[0]
    if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
        N += config["pts_eikonal"].shape[0]
    if "pts_surface" in config and loss_weights.get("curvature", 0.0) > 0:
        N += config["pts_surface"].shape[0]

    JTJ_sum = torch.zeros(N, N, dtype=torch.float64)  # Initialize an empty sum

    for layer_name in model.params:
        if 'weight' in layer_name or 'bias' in layer_name:
            # Initialize the Jacobian blocks for the current layer
            J_blocks = []
            layer_params = {k: v for k, v in model.params.items() if layer_name in k}

            # Interface
            if "pts_boundary" in config and loss_weights.get("interface", 0.0) > 0:
                pts = config["pts_boundary"]
                N = pts.shape[0]
                dr = flatten_layer_grads(model.grad_theta_f(layer_params, pts), layer_name)
                J_blocks.append(np.sqrt(loss_weights["interface"] / N) * dr)

            # Eikonal
            if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
                pts = config["pts_eikonal"]
                N = pts.shape[0]
                dr = flatten_layer_grads(model.grad_theta_r_eikonal(layer_params, pts), layer_name)
                J_blocks.append(np.sqrt(loss_weights["eikonal"] / N) * dr)

            # Curvature
            if "pts_surface" in config and loss_weights.get("curvature", 0.0) > 0:
                pts = config["pts_surface"]
                N = pts.shape[0]
                dr = flatten_layer_grads(model.grad_theta_r_mean_curvature(layer_params, pts), layer_name)
                J_blocks.append(np.sqrt(loss_weights["curvature"] / N) * dr)

            # Concatenate all blocks of J for the current layer and compute JTJ
            J = torch.cat(J_blocks, dim=0).T
            JTJ_layer = J.T @ J

            JTJ_sum += JTJ_layer

    return JTJ_sum


def compute_Jv(model, config, v):
    def flatten_layer_grads(grad_dict, layer_name):
        # Dynamically extract keys corresponding to the given layer
        layer_keys = [k for k in grad_dict if layer_name in k]
        return torch.cat([grad_dict[k].flatten(start_dim=1) for k in layer_keys], dim=1)

    loss_weights = config["loss_weights"]
    params = model.params

    # Initialize the list to store the Jacobian-vector product per layer
    Jv_blocks = []

    # Loop over each layer in the model
    for layer_name in model.params:
        if 'weight' in layer_name or 'bias' in layer_name:
            # Initialize the Jacobian for the current layer
            J_layer = []
            layer_params = {k: v for k, v in model.params.items() if layer_name in k}

            # Interface
            if "pts_boundary" in config and loss_weights.get("interface", 0.0) > 0:
                pts = config["pts_boundary"]
                N = pts.shape[0]
                grad = flatten_layer_grads(model.grad_theta_f(layer_params, pts), layer_name)
                J_layer.append(np.sqrt(loss_weights["interface"] / N) * grad)

            # Eikonal
            if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
                pts = config["pts_eikonal"]
                N = pts.shape[0]
                grad = flatten_layer_grads(model.grad_theta_r_eikonal(layer_params, pts), layer_name)
                J_layer.append(np.sqrt(loss_weights["eikonal"] / N) * grad)

            # Curvature
            if "pts_surface" in config and loss_weights.get("curvature", 0.0) > 0:
                pts = config["pts_surface"]
                N = pts.shape[0]
                grad = flatten_layer_grads(model.grad_theta_r_mean_curvature(layer_params, pts), layer_name)
                J_layer.append(np.sqrt(loss_weights["curvature"] / N) * grad)

            # Stack the Jacobian-vector products for the current layer
            Jv_blocks.append(torch.cat(J_layer, dim=0).T@v)

    # Concatenate all Jacobian-vector products across layers to form the final vector
    Jv_result = torch.cat(Jv_blocks, dim=0)

    return Jv_result