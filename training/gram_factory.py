import torch
import numpy as np

def compute_JTJ_per_residual(params, res_term) -> torch.Tensor:
    """Returns a tensor of shape [P,P] where P is the number of parameters"""
    J_dict = res_term.grad_theta_r(params)
    J = torch.cat([p.flatten(start_dim=1) for p in J_dict.values()], dim=1)
    JTJ = torch.einsum('bi,bj->ij', J, J) / len(res_term.points)
    return JTJ.detach()

def compute_JTJ(params, res_terms) -> torch.Tensor:
    """Returns a tensor of shape [P,P] where P is the number of parameters"""
    ## skip a residual term if it has no points, since these are sometimes found dynamically
    JTJ = sum(
        res_term.weight * compute_JTJ_per_residual(params, res_term)
        for res_term in res_terms.values() if len(res_term.points)>0
    )
    return JTJ

@torch.no_grad()
def flatten_layer_grads(grad_dict, layer_name):
        # Dynamically extract keys corresponding to the given layer
        layer_keys = [k for k in grad_dict if layer_name in k]  
        return torch.cat([grad_dict[k].flatten(start_dim=1) for k in layer_keys], dim=1)

@torch.no_grad()
def compute_JJT(params, res_terms):
    """
    Computes JJT (not JTJ) for Gauss-Newton, iterating over layers and residuals.
    """
    # Determine total number of residuals (N)
    N = sum(len(res_term.points) for res_term in res_terms.values() if len(res_term.points) > 0)
    JJT_sum = torch.zeros(N, N, dtype=torch.float64)

    # For each layer (weight/bias)
    for layer_name in params:
        if 'weight' in layer_name or 'bias' in layer_name:
            J_blocks = []
            layer_params = {k: v for k, v in params.items() if layer_name in k}

            # For each residual term
            for res_term in res_terms.values():
                if len(res_term.points) == 0:
                    continue
                grad_dict = res_term.grad_theta_r(layer_params)
                dr = flatten_layer_grads(grad_dict, layer_name)
                dr = np.sqrt(res_term.weight / len(res_term.points)) * dr
                J_blocks.append(dr)

            if J_blocks:
                J = torch.cat(J_blocks, dim=0)
                JJT_sum += J @ J.T

    return JJT_sum

@torch.no_grad()
def compute_residual(params, res_terms):
    """
    Computes the full residual vector
    """
    r_blocks = []

    for res_term in res_terms.values():
        if len(res_term.points) == 0:
            continue
        # Compute residual for this term
        r = res_term.eval(params).squeeze(1)
        # Weight and scale
        r = np.sqrt(res_term.weight / len(res_term.points)) * r
        r_blocks.append(r)

    # Stack all residuals
    r = torch.cat(r_blocks, dim=0)
    return r.detach()

@torch.no_grad()
def compute_JTv(params, res_terms, v):
    """
    Computes J^T v efficiently, iterating over layers and residual terms.
    """
    JTv_blocks = []

    for layer_name in params:
        if 'weight' in layer_name or 'bias' in layer_name:
            J_layer = []
            layer_params = {k: val for k, val in params.items() if layer_name in k}

            for res_term in res_terms.values():
                if len(res_term.points) == 0:
                    continue
                grad_dict = res_term.grad_theta_r(layer_params)
                grad = flatten_layer_grads(grad_dict, layer_name)  # [N_i, P_layer]
                grad = np.sqrt(res_term.weight / len(res_term.points)) * grad
                J_layer.append(grad)

            if J_layer:
                J = torch.cat(J_layer, dim=0)  # [N, P_layer]
                JTv_blocks.append(J.T @ v)     # [P_layer]

    JTv = torch.cat(JTv_blocks, dim=0)  # [total_P]
    return JTv