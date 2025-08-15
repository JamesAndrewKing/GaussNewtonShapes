import torch

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