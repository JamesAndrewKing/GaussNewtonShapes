import torch
from torch.func import vmap, jacrev, jacfwd, functional_call

_model = None

def bind_model(model):
    """Bind the model once for use in all residuals."""
    global _model
    _model = model

def adjugate_3x3(A):
    # A: tensor of shape (B, 3, 3)
    a = A[:, 0, 0]; b = A[:, 0, 1]; c = A[:, 0, 2]
    d = A[:, 1, 0]; e = A[:, 1, 1]; f = A[:, 1, 2]
    g = A[:, 2, 0]; h = A[:, 2, 1]; i = A[:, 2, 2]

    adj = torch.stack([
        torch.stack([e*i - f*h, c*h - b*i, b*f - c*e], dim=-1),
        torch.stack([f*g - d*i, a*i - c*g, c*d - a*f], dim=-1),
        torch.stack([d*h - e*g, b*g - a*h, a*e - b*d], dim=-1)
    ], dim=1)

    return adj

# ===== Base functions =====

def _f(params, x):
    """Model output (single input)."""
    return functional_call(_model, params, x.unsqueeze(0)).squeeze(0)

def f(params, x):
    """Model output (batched)."""
    return vmap(_f, in_dims=(None, 0))(params, x)

def _grad_x_f(params, x):
    return jacrev(_f, argnums=1)(params, x)

def grad_x_f(params, x):
    return vmap(_grad_x_f, in_dims=(None, 0))(params, x)

def _hess_x_f(params, x):
    return jacfwd(_grad_x_f, argnums=1)(params, x)

def hess_x_f(params, x):
    return vmap(_hess_x_f, in_dims=(None, 0))(params, x)


# ===== Residual functions =====
# r_example returns residual vector of form (N,1)
# grad_theta_r_example returns dict with derivatives per layer of form (N,1,layer_out,layer_in)

def _r_data(params, x, target):
    return (_f(params, x).squeeze() - target).unsqueeze(-1)

def r_data(params, x, target):
    return vmap(_r_data, in_dims=(None, 0, 0), out_dims=(0))(params, x, target)

def _grad_theta_r_data(params, x, target):
    return jacrev(_r_data, argnums=0)(params, x, target)

def grad_theta_r_data(params, x, target):
    return vmap(_grad_theta_r_data, in_dims=(None, 0, 0), out_dims=(0))(params, x, target)

def _r_eikonal(params, x):
    return _grad_x_f(params, x).squeeze(1).square().sum(1).sqrt() - 1

def r_eikonal(params, x):
    return vmap(_r_eikonal, in_dims=(None, 0), out_dims=(0))(params, x)

def _grad_theta_r_eikonal(params, x):
    return jacrev(_r_eikonal, argnums=0)(params, x)

def grad_theta_r_eikonal(params, x):
    return vmap(_grad_theta_r_eikonal, in_dims=(None, 0), out_dims=(0))(params, x)

def _r_normal(params, x, target_normal):
    grad_f = _grad_x_f(params, x).squeeze(1)
    norm_grad_f = grad_f.norm(p=2)
    model_normal = grad_f / norm_grad_f
    return (torch.dot(model_normal.squeeze(0), target_normal)-1).unsqueeze(-1)

def r_normal(params, x, target_normal):
    return vmap(_r_normal, in_dims=(None, 0, 0), out_dims=(0))(params, x, target_normal)

def _grad_theta_r_normal(params, x, target_normal):
    return jacrev(_r_normal, argnums=0)(params, x, target_normal)

def grad_theta_r_normal(params, x, target_normal):
    return vmap(_grad_theta_r_normal, in_dims=(None, 0, 0), out_dims=(0))(params, x, target_normal)

def _r_laplacian(params, x):
    hess_f = _hess_x_f(params, x).squeeze(1)
    tr_hess = torch.einsum('bii->b', hess_f)
    return tr_hess

def r_laplacian(params, x):
    return vmap(_r_laplacian, in_dims=(None, 0), out_dims=(0))(params, x)

def _grad_theta_r_laplacian(params, x):
    return jacrev(_r_laplacian, argnums=0)(params, x)

def grad_theta_r_laplacian(params, x):
    return vmap(_grad_theta_r_laplacian, in_dims=(None, 0), out_dims=(0))(params, x)

def _r_mean_curvature(params, x):
    grad_f = _grad_x_f(params, x).squeeze(1)
    hess_f = _hess_x_f(params, x).squeeze(1)
    grad_hess_grad = torch.einsum('bi,bij,bj->b', grad_f, hess_f, grad_f)
    tr_hess = torch.einsum('bii->b', hess_f)
    norm_grad_f = grad_f.square().sum(1).sqrt()
    mean_curvatures = -(grad_hess_grad - norm_grad_f.pow(2) * tr_hess) / (2 * norm_grad_f.pow(3))
    return mean_curvatures

def r_mean_curvature(params, x):
    return vmap(_r_mean_curvature, in_dims=(None, 0), out_dims=(0))(params, x)

def _grad_theta_r_mean_curvature(params, x):
    return jacrev(_r_mean_curvature, argnums=0)(params, x)

def grad_theta_r_mean_curvature(params, x):
    return vmap(_grad_theta_r_mean_curvature, in_dims=(None, 0), out_dims=(0))(params, x)

def _r_gauss_curvature(params, x):
    grad_f = _grad_x_f(params, x).squeeze(1)
    hess_f = _hess_x_f(params, x).squeeze(1)
    adj_hess_f = adjugate_3x3(hess_f)
    grad_adj_grad = torch.einsum('bi,bij,bj->b', grad_f, adj_hess_f, grad_f)
    norm_grad_f = grad_f.square().sum(1).sqrt()
    gauss_curvature = grad_adj_grad / norm_grad_f.pow(4)
    return gauss_curvature

def r_gauss_curvature(params, x):
    return vmap(_r_gauss_curvature, in_dims=(None, 0), out_dims=(0))(params, x)

def _grad_theta_r_gauss_curvature(params, x):
    return jacrev(_r_gauss_curvature, argnums=0)(params, x)

def grad_theta_r_gauss_curvature(params, x):
    return vmap(_grad_theta_r_gauss_curvature, in_dims=(None, 0), out_dims=(0))(params, x)

def _r_principle_curvature_1(params, x):
    k_m = _r_mean_curvature(params, x)
    k_g = _r_gauss_curvature(params, x)
    k_1 = k_m + torch.sqrt(k_m.pow(2) - k_g)
    return k_1

def r_principle_curvature_1(params, x):
    return vmap(_r_principle_curvature_1, in_dims=(None, 0), out_dims=(0))(params, x)

def _grad_theta_r_principle_curvature_1(params, x):
    return jacrev(_r_principle_curvature_1, argnums=0)(params, x)

def grad_theta_r_principle_curvature_1(params, x):
    return vmap(_grad_theta_r_principle_curvature_1, in_dims=(None, 0), out_dims=(0))(params, x)

def _r_principle_curvature_2(params, x):
    k_m = _r_mean_curvature(params, x)
    k_g = _r_gauss_curvature(params, x)
    k_1 = k_m - torch.sqrt(k_m.pow(2) - k_g)
    return k_1

def r_principle_curvature_2(params, x):
    return vmap(_r_principle_curvature_2, in_dims=(None, 0), out_dims=(0))(params, x)

def _grad_theta_r_principle_curvature_2(params, x):
    return jacrev(_r_principle_curvature_2, argnums=0)(params, x)

def grad_theta_r_principle_curvature_2(params, x):
    return vmap(_grad_theta_r_principle_curvature_2, in_dims=(None, 0), out_dims=(0))(params, x)



