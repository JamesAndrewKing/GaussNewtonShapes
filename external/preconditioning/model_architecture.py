import torch
from torch import nn
from torch import tanh
from torch.func import vmap, jacrev, jacfwd, functional_call

class GeneralNet(nn.Module):
    def __init__(self, ks, act=torch.tanh):
        super(GeneralNet, self).__init__()
        self.ks = ks
        self.fcs = nn.ModuleList([
            nn.Linear(in_features, out_features, dtype=torch.float64)
            for in_features, out_features in zip(self.ks[:-1], self.ks[1:])
        ])
        self.D = len(self.fcs)
        self.act = act
        self.params = dict(self.named_parameters())

    def forward(self, x, z=None):
        if z is not None:
            x = torch.cat([x, z], dim=-1).to(dtype=torch.float64)
        x = self.fcs[0](x)
        for i in range(2, self.D + 1):
            x = self.fcs[i - 1](self.act(x))
        return x

    def f(self, params, x):
        """Wrapper for functional_call."""
        return functional_call(self, params, x.to(dtype=torch.float64))

    def _f_x(self, params, x):
        """Jacobian of f with respect to input x (non-vectorized, private)."""
        return jacrev(self.f, argnums=1)(params, x.to(dtype=torch.float64))

    def vf_x(self, params, x):
        """Vectorized Jacobian of f with respect to input x."""
        return vmap(self._f_x, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _f_xx(self, params, x):
        """Hessian of f with respect to input x (non-vectorized, private)."""
        return jacfwd(self._f_x, argnums=1)(params, x.to(dtype=torch.float64))

    def vf_xx(self, params, x):
        """Vectorized Hessian of f with respect to input x."""
        return vmap(self._f_xx, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _f_laplace(self, params, x):
        """Laplacian of f with respect to input x (non-vectorized, private)."""
        hessian = self._f_xx(params, x).squeeze(1)  # Compute the Hessian
        laplacian = torch.einsum('bii->b', hessian)  # Sum of the diagonal elements
        return laplacian

    def v_f_laplace(self, params, x):
        """Vectorized Laplacian of f."""
        return vmap(self._f_laplace, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _f_mean_curvature(self, params, x):
        """Mean curvature of f with respect to input x (non-vectorized, private)."""
        F = self._f_x(params, x).squeeze(1)
        H = self._f_xx(params, x).squeeze(1)
        ## Quadratic form
        FHFT = torch.einsum('bi,bij,bj->b', F, H, F)
        ## Trace of Hessian
        trH = torch.einsum('bii->b', H)
        ## Norm of gradient
        N = F.square().sum(1).sqrt()
        ## Mean-curvature
        mean_curvatures = -(FHFT - N.pow(2)*trH) / (2*N.pow(3))
        return mean_curvatures

    def v_f_mean_curvature(self, params, x):
        """Vectorized mean curvature of f."""
        return vmap(self._f_mean_curvature, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))
    
    def _d_theta_f_mean_curvature(self, params, x):
        return jacrev(self._f_mean_curvature, argnums=0)(params, x.to(dtype=torch.float64))

    def v_d_theta_f_mean_curvature(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._d_theta_f_mean_curvature, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _phi(self, params, x):
        """Gradient of f with respect to parameters theta (non-vectorized, private)."""
        return jacrev(self.f, argnums=0)(params, x.to(dtype=torch.float64))

    def vphi(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._phi, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _phi_x(self, params, x):
        """Gradient of phi with respect to input x (non-vectorized, private)."""
        return jacfwd(self._phi, argnums=1)(params, x.to(dtype=torch.float64))

    def v_phi_x(self, params, x):
        """Vectorized gradient of phi with respect to input x."""
        return vmap(self._phi_x, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _phi_laplace(self, params, x):
        """Laplacian of phi (non-vectorized, private)."""
        return jacrev(self._f_laplace, argnums=0)(params, x.to(dtype=torch.float64))

    def v_phi_laplace(self, params, x):
        """Vectorized Laplacian of phi."""
        return vmap(self._phi_laplace, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))
