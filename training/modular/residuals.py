"""
### Gauss-Newton modular implementation

Key idea:
- `ResidualLibrary` implements the different residuals, e.g., data, eikonal ...
- `ResidualTerm` implements standard transformations to apply the residuals, e.g., vectorization, Jacobian ...
- `res_term` dictionary registers the residuals that define the specific optimization problem
- `GaussNewton` and its subroutines become simpler since all residuals share the same abstraction
"""

import torch
from torch.func import vmap, jacrev, jacfwd, functional_call
from typing import Callable, Union, Optional

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
# def adjugate_3x3(A):
#     print(A.shape)
#     a, b, c = A[:,0]
#     d, e, f = A[:,1]
#     g, h, i = A[:,2]
#     adj=torch.tensor([
#             [e*i-f*h, c*h-b*i, b*f-c*e],
#             [f*g-d*i, a*i-c*g, c*d-a*f],
#             [d*h-e*g, b*g-a*h, a*e-b*d],
#         ], dtype=A.dtype, device=A.device)  # shape: (3,3,B)
#     return adj.permute(2,0,1)  # reshape to (B,3,3)


def _mean_curv(g, h):
    ## Compute the mean curvature from the Jacobian g and the Hessian h
    ## Static, allows to reuse g, h in the same parent function saving compute
    gHg = torch.einsum('bi,bij,bj->b', g, h, g)
    tr_h = torch.einsum('bii->b', h)
    norm_g = g.square().sum(1).sqrt()
    return -(gHg - norm_g**2 * tr_h) / (2 * norm_g**3)

def _gauss_curv(g, h):
    ## Compute the Gauss curvature from the Jacobian g and the Hessian h
    ## Static, allows to reuse g, h in the same parent function saving compute
    adj = adjugate_3x3(h)
    gAdjg = torch.einsum('bi,bij,bj->b', g, adj, g)
    return gAdjg / g.square().sum(1).pow(2)


class ResidualLibrary:
    r"""
    Implements different residuals of interest amenable to Gauss-Newton: 0.5*\int r(x)^2 dx.
    All residuals share the signature: self, params, point, val -> scalar tensor.
    where val may or may not be used internally as an optional target.
    """
    def __init__(self, model):
        self.model = model

        ## all of the below have the signature 
        ## self.*(params, x) with x being [1, D] for self._func and [B, D] for self.func
        
        ## Single
        self._f        = lambda params, x: functional_call(self.model, params, x.unsqueeze(0)).squeeze(0)
        self._grad_x_f = jacrev(self._f, argnums=1)
        self._hess_x_f = jacfwd(self._grad_x_f, argnums=1)

    def _data(self, params, x, val):
        return (self._f(params, x).squeeze() - val).unsqueeze(-1) ## TODO: do we need the (un)squeezing?
    
    def _design_region(self, params, x, val=None):
        return torch.minimum(torch.tensor([0], dtype=x.dtype), self._f(params, x).squeeze()).unsqueeze(-1)
    
    def _connectedness(self, params, x, val=None):
        return torch.minimum(torch.tensor([0], dtype=x.dtype), -self._f(params, x).squeeze()).unsqueeze(-1)

    def _eikonal(self, params, x, val=None):
        return self._grad_x_f(params, x).squeeze(1).square().sum(1).sqrt() - 1 ## we could treat val=1

    def _normal(self, params, x, true_normal):
        g = self._grad_x_f(params, x).squeeze(1)
        pred_normal = g / g.norm(p=2)
        return (torch.dot(pred_normal.squeeze(0), true_normal) - 1).unsqueeze(-1)

    def _laplacian(self, params, x, val=None):
        h = self._hess_x_f(params, x).squeeze(1)
        return torch.einsum('bii->b', h)

    def _mean_curvature(self, params, x, val=None):
        g = self._grad_x_f(params, x).squeeze(1)
        h = self._hess_x_f(params, x).squeeze(1)
        return _mean_curv(g, h)
    
    def _gauss_curvature(self, params, x, val=None):
        g = self._grad_x_f(params, x).squeeze(1)
        h = self._hess_x_f(params, x).squeeze(1)
        return _gauss_curv(g, h)

    def _principal_curvature_1(self, params, x, val=None):
        g = self._grad_x_f(params, x).squeeze(1)
        h = self._hess_x_f(params, x).squeeze(1)
        k_m = _mean_curv(g, h)
        k_g = _gauss_curv(g, h)
        return k_m + torch.sqrt(k_m**2 - k_g)

    def _principal_curvature_2(self, params, x, val=None):
        g = self._grad_x_f(params, x).squeeze(1)
        h = self._hess_x_f(params, x).squeeze(1)
        k_m = _mean_curv(g, h)
        k_g = _gauss_curv(g, h)
        return k_m - torch.sqrt(k_m**2 - k_g)
    
    def _strain(self, params, x, val=None):
        r"""
        Strain is 
        \int k1^2 + k2^2 dx = \int \sqrt{k1^2 + k2^2}^2 dx
        so the residual can be written as r(x) = \sqrt{k1^2 + k2^2}.
        This can be simplified in terms of the mean and Gaussian curvatures:
        k1^2 + k2^2 = 4*k_m^2 - 2*k_g
        """
        g = self._grad_x_f(params, x).squeeze(1)
        h = self._hess_x_f(params, x).squeeze(1)
        k_m = _mean_curv(g, h)
        k_g = _gauss_curv(g, h)
        return torch.sqrt(4*k_m**2 - 2*k_g)
    
    
class ResidualTerm:
    def __init__(
        self, 
        func: Callable, 
        weight: float, 
        points: torch.Tensor, 
        vals: Optional[Union[float, int, torch.Tensor, None]] = None
    ):
        self.func = func
        self.weight = weight
        self.points = points
        self.vals = vals
        
    def vmap(self, func: Callable) -> Callable:
        """
        Vectorizes the callable func(params, points, vals) based on the format of self.vals.
        This is used for both the residual itself and its parameter Jacobian.
        """
        ## Same value for all points (including None).
        ## More scalar cases exist, eg 0-dim torch.Tensor, numpy.ndarray but hard to check for a general scalar-like
        if self.vals is None or isinstance(self.vals, (float, int)):
            return vmap(func, in_dims=(None, 0, None))
        ### Unique value for each point 
        elif isinstance(self.vals, torch.Tensor) and len(self.vals)==len(self.points):
            return vmap(func, in_dims=(None, 0, 0))
        else:
            raise ValueError(f"Invalid format for vals: {type(self.vals)}, shape: {getattr(self.vals, 'shape', None)}")
    
    def eval(self, params) -> torch.Tensor:
        """
        Evaluate the model on the batch of self.points using vmap.
        Returns:
            Tensor of shape [B, 1]: one residual value per point
        """
        return self.vmap(self.func)(params, self.points, self.vals)

    def unweighted_loss(self, params) -> torch.Tensor:
        r"""
        Evaluate the residual 0.5 * \int r(x)^2 dx
        Returns:
            Scalar tensor of shape []
        """
        return 0.5*self.eval(params).square().mean()
    
    def weighted_loss(self, params) -> torch.Tensor:
        """
        Returns:
            Scalar tensor of shape []
        """
        return self.weight * self.unweighted_loss(params)
    
    def grad_theta_r(self, params) -> torch.Tensor:
        """
        Evaluate the Jacobian of the residual on the batch of self.points using vmap.
        Returns:
            dict of Jacobian tensors, for each param tensor the shape is: [B, 1, *param.shape]
        """
        _grad_theta_r = jacrev(self.func, argnums=0)
        return self.vmap(_grad_theta_r)(params, self.points, self.vals) 
    
    
def compute_loss(params, res_terms, return_unweighted_losses=False):
    unweighted_losses = {key: res_term.unweighted_loss(params) for key, res_term in res_terms.items()} ## might be used for logging
    loss = sum(res_terms[key].weight*unweighted_losses[key] for key in res_terms)
    if return_unweighted_losses:
        return loss, unweighted_losses
    return loss