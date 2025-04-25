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
    
    if loss_weights.get("mean_curvature", 0) != 0:
        A += loss_weights["mean_curvature"] * calculate_A_mean_curvature(model, pts_surface)

    return A.detach()

# def calculate_A_interface(model, pts):
#     dr_dict = model.grad_theta_f(model.params, pts)
#     dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
#     A_interface = torch.einsum('bi,bj->ij', dr, dr) / len(pts)
#     return A_interface.detach()

# def calculate_A_eikonal(model, pts):
#     dr_dict = model.grad_theta_r_eikonal(model.params, pts)
#     dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
#     A_eikonal = torch.einsum('bi,bj->ij', dr, dr) / len(pts)
#     return A_eikonal

# def calculate_A_mean_curvature(model, pts, vals_mean_curvature):
#     dr_dict = model.grad_theta_r_mean_curvature(model.params, pts, vals_mean_curvature)
#     dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
#     A_mean_curvature = torch.einsum('bi,bj->ij', dr, dr) / len(pts)
#     return A_mean_curvature

# def calculate_A_laplacian(model, pts):
#     dr_dict = model.grad_theta_r_laplacian(model.params, pts)
#     dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
#     A_laplacian = torch.einsum('bi,bj->ij', dr, dr) / len(pts)
#     return A_laplacian
    

# def compute_gram_matrix(model, config):
#     pts_boundary = config.get("pts_boundary")
#     pts_eikonal = config.get("pts_eikonal")
#     pts_surface = config.get("pts_surface")
#     vals_mean_curvature = config.get("vals_mean_curvature")
#     loss_weights = config.get("loss_weights")

#     A = torch.zeros((model.num_params, model.num_params), dtype=torch.float64)

#     if loss_weights.get("interface", 0) != 0:
#         A += loss_weights["interface"] * calculate_A_interface(model, pts_boundary)
    
#     if loss_weights.get("eikonal", 0) != 0:
#         A += loss_weights["eikonal"] * calculate_A_eikonal(model, pts_eikonal)
    
#     if loss_weights.get("mean_curvature", 0) != 0:
#         A += loss_weights["mean_curvature"] * calculate_A_mean_curvature(model, pts_surface, vals_mean_curvature)

#     if loss_weights.get("laplacian", 0) != 0:
#         A += loss_weights["laplacian"] * calculate_A_laplacian(model, pts_surface)

#     return A.detach()




# Old version that flattened out the whole parameter vector:
# @torch.no_grad()
# def compute_jacobian_and_residual(model, config):
#     def flatten_grad_dict(grad_dict):
#         return torch.cat([p.flatten(start_dim=1) for p in grad_dict.values()], dim=1)

#     loss_weights = config["loss_weights"]
#     params = model.params

#     J_blocks = []
#     r_blocks = []

#     # Interface
#     if "pts_boundary" in config and loss_weights.get("interface", 0.0) > 0:
#         pts = config["pts_boundary"]
#         N = pts.shape[0]
#         r = model.f(params, pts).squeeze(1)
#         r_blocks.append(
#             np.sqrt(loss_weights["interface"] / N) * r
#         )
#         dr = flatten_grad_dict(model.grad_theta_f(params, pts))
#         J_blocks.append(
#             np.sqrt(loss_weights["interface"] / N) * dr
#         )

#     # Eikonal
#     if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
#         pts = config["pts_eikonal"]
#         N = pts.shape[0]
#         r = model.r_eikonal(params, pts).squeeze(1)
#         r_blocks.append(
#             np.sqrt(loss_weights["eikonal"] / N) * r
#         )
#         dr = flatten_grad_dict(model.grad_theta_r_eikonal(params, pts))
#         J_blocks.append(
#             np.sqrt(loss_weights["eikonal"] / N) * dr
#         )

#     # Curvature
#     if "pts_surface" in config and loss_weights.get("curvature", 0.0) > 0:
#         pts = config["pts_surface"]
#         N = pts.shape[0]
#         r = model.r_mean_curvature(params, pts).squeeze(1)
#         r_blocks.append(
#             np.sqrt(loss_weights["curvature"] / N) * r
#         )
#         dr = flatten_grad_dict(model.grad_theta_r_mean_curvature(params, pts))
#         J_blocks.append(
#             np.sqrt(loss_weights["curvature"] / N) * dr
#         )

#     # Laplacian
#     if "pts_surface" in config and loss_weights.get("laplacian", 0.0) > 0:
#         pts = config["pts_surface"]
#         N = pts.shape[0]
#         r = model.r_laplacian(params, pts).squeeze(1)
#         r_blocks.append(
#             np.sqrt(loss_weights["laplacian"] / N) * r
#         )
#         dr = flatten_grad_dict(model.grad_theta_r_laplacian(params, pts))
#         J_blocks.append(
#             np.sqrt(loss_weights["laplacian"] / N) * dr
#         )

#     # Stack full residual and Jacobian
#     r = torch.cat(r_blocks, dim=0)
#     J = torch.cat(J_blocks, dim=0).T

#     return J.detach(), r.detach()
@torch.no_grad()
def flatten_layer_grads(grad_dict, layer_name):
        # Dynamically extract keys corresponding to the given layer
        layer_keys = [k for k in grad_dict if layer_name in k]  
        return torch.cat([grad_dict[k].flatten(start_dim=1) for k in layer_keys], dim=1)

@torch.no_grad()
def compute_residual(model, config):
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

    # Data
    if "data" in config and loss_weights.get("data", 0.0) > 0:
        data = config["data"]
        pts, vals = data["pts_data"], data["vals_data"]
        N = pts.shape[0]
        r = model.r_data(params, pts, vals)
        r_blocks.append(
            np.sqrt(loss_weights["data"] / N) * r
        )

    # Eikonal
    if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
        pts = config["pts_eikonal"]
        N = pts.shape[0]
        r = model.r_eikonal(params, pts).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["eikonal"] / N) * r
        )

    # Normal
    if "pts_surface" in config and loss_weights.get("normal", 0.0) > 0:
        pts = config["pts_surface"]
        N = pts.shape[0]
        true_normals = config["true_normals"]
        r = model.r_normal(params, pts, true_normals).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["normal"] / N) * r
        )
        
    # Mean Curvature
    if "pts_surface" in config and loss_weights.get("mean_curvature", 0.0) > 0:
        pts = config["pts_surface"]
        N = pts.shape[0]
        vals = config.get("vals_mean_curvature", torch.zeros(N, dtype=torch.float64))
        r = model.r_mean_curvature(params, pts, vals).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["mean_curvature"] / N) * r
        )

    # Gauss Curvature
    if "pts_surface" in config and loss_weights.get("gauss_curvature", 0.0) > 0:
        pts = config["pts_surface"]
        vals = config["vals_gauss_curvature"]
        N = pts.shape[0]
        r = model.r_gauss_curvature(params, pts, vals).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["gauss_curvature"] / N) * r
        )

    # Laplacian
    if "pts_surface" in config and loss_weights.get("laplacian", 0.0) > 0:
        pts = config["pts_surface"]
        N = pts.shape[0]
        r = model.r_laplacian(params, pts).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["laplacian"] / N) * r
        )

    # Surface Strain
    if "pts_surface" in config and loss_weights.get("surface_strain", 0.0) > 0:
        pts = config["pts_surface"]
        N = pts.shape[0]
        r = model.r_principle_curvature_1(params, pts).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["surface_strain"] / N) * r
        )
        r = model.r_principle_curvature_2(params, pts).squeeze(1)
        r_blocks.append(
            np.sqrt(loss_weights["surface_strain"] / N) * r
        )

    # Stack full residual and Jacobian
    r = torch.cat(r_blocks, dim=0)

    return r.detach()

@torch.no_grad()
def compute_JTJ(model, config):

    loss_weights = config["loss_weights"]
    params = model.params

    # Calculate size of residual vector
    N = 0
    if "pts_boundary" in config and loss_weights.get("interface", 0.0) > 0:
        N += config["pts_boundary"].shape[0]
    if "data" in config and loss_weights.get("data", 0.0) > 0:
        N += config["data"]["pts_data"].shape[0]
    if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
        N += config["pts_eikonal"].shape[0]
    if "pts_surface" in config and loss_weights.get("normal", 0.0) > 0:
        N += config["pts_surface"].shape[0]
    if "pts_surface" in config and loss_weights.get("mean_curvature", 0.0) > 0:
        N += config["pts_surface"].shape[0]
    if "pts_surface" in config and loss_weights.get("gauss_curvature", 0.0) > 0:
        N += config["pts_surface"].shape[0]
    if "pts_surface" in config and loss_weights.get("laplacian", 0.0) > 0:
        N += config["pts_surface"].shape[0]
    if "pts_surface" in config and loss_weights.get("surface_strain", 0.0) > 0:
        N += 2*config["pts_surface"].shape[0]

    JTJ_sum = torch.zeros(N, N, dtype=torch.float64)  # Initialize an empty sum

    for layer_name in model.params:
        if 'weight' in layer_name or 'bias' in layer_name:
            J_blocks = []
            layer_params = {k: v for k, v in model.params.items() if layer_name in k}

            # Interface
            if "pts_boundary" in config and loss_weights.get("interface", 0.0) > 0:
                pts = config["pts_boundary"]
                N = pts.shape[0]
                dr = flatten_layer_grads(model.grad_theta_f(layer_params, pts), layer_name)
                J_blocks.append(np.sqrt(loss_weights["interface"] / N) * dr)

            # Data
            if "data" in config and loss_weights.get("data", 0.0) > 0:
                data = config["data"]
                pts, vals = data["pts_data"], data["vals_data"]
                N = pts.shape[0]
                dr = flatten_layer_grads(model.grad_theta_r_data(layer_params, pts, vals), layer_name)
                J_blocks.append(np.sqrt(loss_weights["data"] / N) * dr)

            # Eikonal
            if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
                pts = config["pts_eikonal"]
                N = pts.shape[0]
                dr = flatten_layer_grads(model.grad_theta_r_eikonal(layer_params, pts), layer_name)
                J_blocks.append(np.sqrt(loss_weights["eikonal"] / N) * dr)

            # Normal
            if "pts_surface" in config and loss_weights.get("normal", 0.0) > 0:
                pts = config["pts_surface"]
                true_normals = config["true_normals"]
                N = pts.shape[0]
                dr = flatten_layer_grads(model.grad_theta_r_normal(layer_params, pts, true_normals), layer_name)
                J_blocks.append(np.sqrt(loss_weights["normal"] / N) * dr)

            # Mean Curvature
            if "pts_surface" in config and loss_weights.get("mean_curvature", 0.0) > 0:
                pts = config["pts_surface"]
                N = pts.shape[0]
                vals = config.get("vals_mean_curvature", torch.zeros(N, dtype=torch.float64))
                dr = flatten_layer_grads(model.grad_theta_r_mean_curvature(layer_params, pts, vals), layer_name)
                J_blocks.append(np.sqrt(loss_weights["mean_curvature"] / N) * dr)

            # Gauss Curvature
            if "pts_surface" in config and loss_weights.get("gauss_curvature", 0.0) > 0:
                pts = config["pts_surface"]
                vals = config["vals_gauss_curvature"]
                N = pts.shape[0]
                dr = flatten_layer_grads(model.grad_theta_r_gauss_curvature(layer_params, pts, vals), layer_name)
                J_blocks.append(np.sqrt(loss_weights["gauss_curvature"] / N) * dr)

            # Laplacian
            if "pts_surface" in config and loss_weights.get("laplacian", 0.0) > 0:
                pts = config["pts_surface"]
                N = pts.shape[0]
                dr = flatten_layer_grads(model.grad_theta_r_laplacian(layer_params, pts), layer_name)
                J_blocks.append(np.sqrt(loss_weights["laplacian"] / N) * dr)

            # Surface Strain
            if "pts_surface" in config and loss_weights.get("surface_strain", 0.0) > 0:
                pts = config["pts_surface"]
                N = pts.shape[0]
                dr = flatten_layer_grads(model.grad_theta_r_principle_curvature_1(layer_params, pts), layer_name)
                J_blocks.append(np.sqrt(loss_weights["surface_strain"] / N) * dr)
                dr = flatten_layer_grads(model.grad_theta_r_principle_curvature_2(layer_params, pts), layer_name)
                J_blocks.append(np.sqrt(loss_weights["surface_strain"] / N) * dr)

            # Concatenate all blocks of J for the current layer and compute JTJ
            J = torch.cat(J_blocks, dim=0).T
            JTJ_sum += J.T @ J

    return JTJ_sum


def compute_JJT(model, config):

    loss_weights = config["loss_weights"]
    params = model.params

    # Initialize JJT as a zero matrix of size (params x params)
    JJT_sum = torch.zeros(model.num_params, model.num_params, dtype=torch.float64)

    # Interface loss
    if "pts_boundary" in config and loss_weights.get("interface", 0.0) > 0:
        pts = config["pts_boundary"]
        dr_dict = model.grad_theta_f(model.params, pts)
        dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
        JJT_sum += loss_weights["interface"] * torch.einsum('bi,bj->ij', dr, dr) / len(pts)

    # Data loss
    if "data" in config and loss_weights.get("data", 0.0) > 0:
        data = config["data"]
        pts, vals = data["pts_data"], data["vals_data"]
        dr_dict = model.grad_theta_r_data(model.params, pts, vals)
        dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
        JJT_sum += loss_weights["data"] * torch.einsum('bi,bj->ij', dr, dr) / len(pts)

    # Eikonal loss
    if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
        pts = config["pts_eikonal"]
        dr_dict = model.grad_theta_r_eikonal(model.params, pts)
        dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
        JJT_sum += loss_weights["eikonal"] * torch.einsum('bi,bj->ij', dr, dr) / len(pts)

    # Normal loss
    if "pts_surface" in config and loss_weights.get("normal", 0.0) > 0:
        pts = config["pts_surface"]
        true_normals = config["true_normals"]
        dr_dict = model.grad_theta_r_normal(model.params, pts, true_normals)
        dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
        JJT_sum += loss_weights["normal"] * torch.einsum('bi,bj->ij', dr, dr) / len(pts)

    # Mean Curvature loss
    if "pts_surface" in config and loss_weights.get("mean_curvature", 0.0) > 0:
        pts = config["pts_surface"]
        vals = config.get("vals_mean_curvature", torch.zeros(pts.shape[0], dtype=torch.float64))
        dr_dict = model.grad_theta_r_mean_curvature(model.params, pts, vals)
        dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
        JJT_sum += loss_weights["mean_curvature"] * torch.einsum('bi,bj->ij', dr, dr) / len(pts)

    # Gauss Curvature loss
    if "pts_surface" in config and loss_weights.get("gauss_curvature", 0.0) > 0:
        pts = config["pts_surface"]
        vals = config["vals_gauss_curvature"]
        dr_dict = model.grad_theta_r_gauss_curvature(model.params, pts, vals)
        dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
        JJT_sum += loss_weights["gauss_curvature"] * torch.einsum('bi,bj->ij', dr, dr) / len(pts)

    # Laplacian loss
    if "pts_surface" in config and loss_weights.get("laplacian", 0.0) > 0:
        pts = config["pts_surface"]
        dr_dict = model.grad_theta_r_laplacian(model.params, pts)
        dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
        JJT_sum += loss_weights["laplacian"] * torch.einsum('bi,bj->ij', dr, dr) / len(pts)

    # Surface Strain loss
    if "pts_surface" in config and loss_weights.get("surface_strain", 0.0) > 0:
        pts = config["pts_surface"]
        dr_dict = model.grad_theta_r_principle_curvature_1(model.params, pts)
        dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
        JJT_sum += loss_weights["surface_strain"] * torch.einsum('bi,bj->ij', dr, dr) / len(pts)
        dr_dict = model.grad_theta_r_principle_curvature_2(model.params, pts)
        dr = torch.cat([p.flatten(start_dim=1) for p in dr_dict.values()], dim=1)
        JJT_sum += loss_weights["surface_strain"] * torch.einsum('bi,bj->ij', dr, dr) / len(pts)

    return JJT_sum.detach()


@torch.no_grad()
def compute_Jv(model, config, v):
    loss_weights = config["loss_weights"]
    params = model.params

    # Initialize the list to store the Jacobian-vector product per layer
    Jv_blocks = []

    for layer_name in model.params:
        if 'weight' in layer_name or 'bias' in layer_name:
            J_layer = []
            layer_params = {k: v for k, v in model.params.items() if layer_name in k}

            # Interface
            if "pts_boundary" in config and loss_weights.get("interface", 0.0) > 0:
                pts = config["pts_boundary"]
                N = pts.shape[0]
                grad = flatten_layer_grads(model.grad_theta_f(layer_params, pts), layer_name)
                J_layer.append(np.sqrt(loss_weights["interface"] / N) * grad)

            # Data
            if "data" in config and loss_weights.get("data", 0.0) > 0:
                data = config["data"]
                pts, vals = data["pts_data"], data["vals_data"]
                N = pts.shape[0]
                grad = flatten_layer_grads(model.grad_theta_r_data(layer_params, pts, vals), layer_name)
                J_layer.append(np.sqrt(loss_weights["data"] / N) * grad)

            # Eikonal
            if "pts_eikonal" in config and loss_weights.get("eikonal", 0.0) > 0:
                pts = config["pts_eikonal"]
                N = pts.shape[0]
                grad = flatten_layer_grads(model.grad_theta_r_eikonal(layer_params, pts), layer_name)
                J_layer.append(np.sqrt(loss_weights["eikonal"] / N) * grad)

            # Normal
            if "pts_surface" in config and loss_weights.get("normal", 0.0) > 0:
                pts = config["pts_surface"]
                true_normals = config["true_normals"]
                N = pts.shape[0]
                grad = flatten_layer_grads(model.grad_theta_r_normal(layer_params, pts, true_normals), layer_name)
                J_layer.append(np.sqrt(loss_weights["normal"] / N) * grad)

            # Mean Curvature
            if "pts_surface" in config and loss_weights.get("mean_curvature", 0.0) > 0:
                pts = config["pts_surface"]
                N = pts.shape[0]
                vals = config.get("vals_mean_curvature", torch.zeros(N, dtype=torch.float64))  
                grad = flatten_layer_grads(model.grad_theta_r_mean_curvature(layer_params, pts, vals), layer_name)
                J_layer.append(np.sqrt(loss_weights["mean_curvature"] / N) * grad)

            # Gauss Curvature
            if "pts_surface" in config and loss_weights.get("gauss_curvature", 0.0) > 0:
                pts = config["pts_surface"]
                vals = config["vals_gauss_curvature"]
                N = pts.shape[0]
                grad = flatten_layer_grads(model.grad_theta_r_gauss_curvature(layer_params, pts, vals), layer_name)
                J_layer.append(np.sqrt(loss_weights["gauss_curvature"] / N) * grad)

            # Laplacian
            if "pts_surface" in config and loss_weights.get("laplacian", 0.0) > 0:
                pts = config["pts_surface"]
                N = pts.shape[0]
                grad = flatten_layer_grads(model.grad_theta_r_laplacian(layer_params, pts), layer_name)
                J_layer.append(np.sqrt(loss_weights["laplacian"] / N) * grad)

            # Surface Strain
            if "pts_surface" in config and loss_weights.get("surface_strain", 0.0) > 0:
                pts = config["pts_surface"]
                N = pts.shape[0]
                grad = flatten_layer_grads(model.grad_theta_r_principle_curvature_1(layer_params, pts), layer_name)
                J_layer.append(np.sqrt(loss_weights["surface_strain"] / N) * grad)
                grad = flatten_layer_grads(model.grad_theta_r_principle_curvature_2(layer_params, pts), layer_name)
                J_layer.append(np.sqrt(loss_weights["surface_strain"] / N) * grad)

            # Append the Jacobian-vector products for the current layer
            Jv_blocks.append(torch.cat(J_layer, dim=0).T@v)

    # Stack all Jacobian-vector products across layers to form the final vector
    Jv_result = torch.cat(Jv_blocks, dim=0)

    return Jv_result