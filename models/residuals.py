import torch
from torch.func import vmap, jacrev, jacfwd, functional_call

def grad_x_f(model, params, x):
    """Jacobian of f with respect to input x."""
    def _grad_x_f(params, x):
        """Jacobian of f with respect to input x (non-vectorized, private)."""
        return jacrev(model.f, argnums=1)(params, x.to(dtype=torch.float64))

    return vmap(_grad_x_f, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))
