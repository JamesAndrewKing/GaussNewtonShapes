import torch
from torch import nn
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

    def _grad_x_f(self, params, x):
        """Jacobian of f with respect to input x (non-vectorized, private)."""
        return jacrev(self.f, argnums=1)(params, x.to(dtype=torch.float64))

    def grad_x_f(self, params, x):
        """Vectorized Jacobian of f with respect to input x."""
        return vmap(self._grad_x_f, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _hess_x_f(self, params, x):
        """Hessian of f with respect to input x (non-vectorized, private)."""
        return jacfwd(self._grad_x_f, argnums=1)(params, x.to(dtype=torch.float64))

    def hess_x_f(self, params, x):
        """Vectorized Hessian of f with respect to input x."""
        return vmap(self._hess_x_f, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _r_mean_curvature(self, params, x):
        grad_f = self._grad_x_f(params, x).squeeze(1)
        hess_f = self._hess_x_f(params, x).squeeze(1)
        grad_hess_grad = torch.einsum('bi,bij,bj->b', grad_f, hess_f, grad_f)
        tr_hess = torch.einsum('bii->b', hess_f)
        norm_grad_f = grad_f.square().sum(1).sqrt()
        mean_curvatures = -(grad_hess_grad - norm_grad_f.pow(2) * tr_hess) / (2 * norm_grad_f.pow(3))
        return mean_curvatures

    def r_mean_curvature(self, params, x):
        return vmap(self._r_mean_curvature, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _grad_theta_r_mean_curvature(self, params, x):
        return jacrev(self._r_mean_curvature, argnums=0)(params, x.to(dtype=torch.float64))

    def grad_theta_r_mean_curvature(self, params, x):
        return vmap(self._grad_theta_r_mean_curvature, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _r_eikonal(self, params, x):
        """Eikonal residual of f with respect to input x (non-vectorized, private)."""
        return self._grad_x_f(params, x).squeeze(1).square().sum(1).sqrt() - 1

    def r_eikonal(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._r_eikonal, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _grad_theta_r_eikonal(self, params, x):
        return jacrev(self._r_eikonal, argnums=0)(params, x.to(dtype=torch.float64))

    def grad_theta_r_eikonal(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._grad_theta_r_eikonal, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _grad_theta_f(self, params, x):
        """Gradient of f with respect to parameters theta (non-vectorized, private)."""
        return jacrev(self.f, argnums=0)(params, x.to(dtype=torch.float64))

    def grad_theta_f(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._grad_theta_f, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))


class BunnyNet(nn.Module):
    def __init__(self, ks, act=torch.sin):
        super(BunnyNet, self).__init__()
        self.ks = ks
        self.fcs = nn.ModuleList([
            nn.Linear(in_features, out_features, dtype=torch.float64)
            for in_features, out_features in zip(self.ks[:-1], self.ks[1:])
        ])
        self.apply(self.init_weights)
        self.D = len(self.fcs)
        self.act = act
        self.params = dict(self.named_parameters())

    def init_weights(self, m):
        """Initialize weights using Xavier initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Initialize biases to zero

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

    def _grad_x_f(self, params, x):
        """Jacobian of f with respect to input x (non-vectorized, private)."""
        return jacrev(self.f, argnums=1)(params, x.to(dtype=torch.float64))

    def grad_x_f(self, params, x):
        """Vectorized Jacobian of f with respect to input x."""
        return vmap(self._grad_x_f, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _hess_x_f(self, params, x):
        """Hessian of f with respect to input x (non-vectorized, private)."""
        return jacfwd(self._grad_x_f, argnums=1)(params, x.to(dtype=torch.float64))

    def hess_x_f(self, params, x):
        """Vectorized Hessian of f with respect to input x."""
        return vmap(self._hess_x_f, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _r_laplacian(self, params, x):
        hess_f = self._hess_x_f(params, x).squeeze(1)
        tr_hess = torch.einsum('bii->b', hess_f)
        return tr_hess

    def r_laplacian(self, params, x):
        return vmap(self._r_mean_curvature, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _grad_theta_r_laplacian(self, params, x):
        return jacrev(self._r_mean_curvature, argnums=0)(params, x.to(dtype=torch.float64))

    def grad_theta_r_laplacian(self, params, x):
        return vmap(self._grad_theta_r_mean_curvature, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _r_eikonal(self, params, x):
        """Eikonal residual of f with respect to input x (non-vectorized, private)."""
        return self._grad_x_f(params, x).squeeze(1).square().sum(1).sqrt() - 1

    def r_eikonal(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._r_eikonal, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _grad_theta_r_eikonal(self, params, x):
        return jacrev(self._r_eikonal, argnums=0)(params, x.to(dtype=torch.float64))

    def grad_theta_r_eikonal(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._grad_theta_r_eikonal, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _grad_theta_f(self, params, x):
        """Gradient of f with respect to parameters theta (non-vectorized, private)."""
        return jacrev(self.f, argnums=0)(params, x.to(dtype=torch.float64))

    def grad_theta_f(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._grad_theta_f, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))
