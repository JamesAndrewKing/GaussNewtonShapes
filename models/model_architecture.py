import torch
from torch import nn
from torch.func import vmap, jacrev, jacfwd, functional_call

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

# The network I'm using for the Mean Curvature Experiments
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

# The network I'm using for everything else, TODO: merge into one, have losses as seperate functions, not model attributes
class PlaceholderNet(nn.Module):
    def __init__(self, ks, act=torch.tanh):
        super(PlaceholderNet, self).__init__()
        self.ks = ks
        self.fcs = nn.ModuleList([
            nn.Linear(in_features, out_features, dtype=torch.float64)
            for in_features, out_features in zip(self.ks[:-1], self.ks[1:])
        ])
        self.apply(self.init_weights)
        self.D = len(self.fcs)
        self.act = act
        self.params = dict(self.named_parameters())
        self.num_params = sum(p.numel() for p in self.parameters())

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

    # def _r_data(self, params, x, target):
    #     return self.f(params, x).squeeze() - target

    # def r_data(self, params, x, target):
    #     return vmap(self._r_data, in_dims=(None, 0, 0), out_dims=(0))(params, x.to(dtype=torch.float64), target)

    # def _grad_theta_r_data(self, params, x, target):
    #     return jacrev(self._r_data, argnums=0)(params, x.to(dtype=torch.float64), target)

    # def grad_theta_r_data(self, params, x, target):
    #     return vmap(self._grad_theta_r_data, in_dims=(None, 0, 0), out_dims=(0))(params, x.to(dtype=torch.float64), target)

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


    def _r_normal(self, params, x, target_normal):
        # grad_f = self._grad_x_f(params, x).squeeze(1)  # Shape: (3,)
        # norm_grad_f = grad_f.norm(p=2)
        # model_normal = grad_f / norm_grad_f
        # return (model_normal - target_normal).norm(p=2).unsqueeze(-1)
        # This normal loss seems to work better:
        grad_f = self._grad_x_f(params, x).squeeze(1)  # Shape: (3,)
        norm_grad_f = grad_f.norm(p=2)
        model_normal = grad_f / norm_grad_f
        # return (model_normal.squeeze(0).T @ target_normal -1).unsqueeze(-1)
        return (torch.dot(model_normal.squeeze(0), target_normal)-1).unsqueeze(-1)

    def r_normal(self, params, x, target_normal):
        return vmap(self._r_normal, in_dims=(None, 0, 0), out_dims=(0))(params, x.to(dtype=torch.float64), target_normal)

    def _grad_theta_r_normal(self, params, x, target_normal):
        return jacrev(self._r_normal, argnums=0)(params, x.to(dtype=torch.float64), target_normal)

    def grad_theta_r_normal(self, params, x, target_normal):
        return vmap(self._grad_theta_r_normal, in_dims=(None, 0, 0), out_dims=(0))(params, x.to(dtype=torch.float64), target_normal)

    # def _r_laplacian(self, params, x):
    #     hess_f = self._hess_x_f(params, x).squeeze(1)
    #     tr_hess = torch.einsum('bii->b', hess_f)
    #     return tr_hess

    # def r_laplacian(self, params, x):
    #     return vmap(self._r_laplacian, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    # def _grad_theta_r_laplacian(self, params, x):
    #     return jacrev(self._r_laplacian, argnums=0)(params, x.to(dtype=torch.float64))

    # def grad_theta_r_laplacian(self, params, x):
    #     return vmap(self._grad_theta_r_laplacian, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _r_mean_curvature(self, params, x, target):
        grad_f = self._grad_x_f(params, x).squeeze(1)
        hess_f = self._hess_x_f(params, x).squeeze(1)
        grad_hess_grad = torch.einsum('bi,bij,bj->b', grad_f, hess_f, grad_f)
        tr_hess = torch.einsum('bii->b', hess_f)
        norm_grad_f = grad_f.square().sum(1).sqrt()
        mean_curvatures = -(grad_hess_grad - norm_grad_f.pow(2) * tr_hess) / (2 * norm_grad_f.pow(3))
        # Clipping for stability, doesnt seem to help
        # clip_param = 1e-3
        # mean_curvatures = torch.tanh(clip_param*mean_curvatures)/clip_param
        return mean_curvatures - target

    def r_mean_curvature(self, params, x, target):
        return vmap(self._r_mean_curvature, in_dims=(None, 0, 0), out_dims=(0))(params, x.to(dtype=torch.float64), target)

    def _grad_theta_r_mean_curvature(self, params, x, target):
        return jacrev(self._r_mean_curvature, argnums=0)(params, x.to(dtype=torch.float64), target)

    def grad_theta_r_mean_curvature(self, params, x, target):
        return vmap(self._grad_theta_r_mean_curvature, in_dims=(None, 0, 0), out_dims=(0))(params, x.to(dtype=torch.float64), target)
    
    def _r_gauss_curvature(self, params, x, target):
        grad_f = self._grad_x_f(params, x).squeeze(1)
        hess_f = self._hess_x_f(params, x).squeeze(1)
        adj_hess_f = adjugate_3x3(hess_f)
        grad_adj_grad = torch.einsum('bi,bij,bj->b', grad_f, adj_hess_f, grad_f)
        norm_grad_f = grad_f.square().sum(1).sqrt()
        gauss_curvature = grad_adj_grad / norm_grad_f.pow(4)
        return gauss_curvature - target

    def r_gauss_curvature(self, params, x, target):
        return vmap(self._r_gauss_curvature, in_dims=(None, 0, 0), out_dims=(0))(params, x.to(dtype=torch.float64), target)
    
    def _grad_theta_r_gauss_curvature(self, params, x, target):
        return jacrev(self._r_gauss_curvature, argnums=0)(params, x.to(dtype=torch.float64), target)

    def grad_theta_r_gauss_curvature(self, params, x, target):
        return vmap(self._grad_theta_r_gauss_curvature, in_dims=(None, 0, 0), out_dims=(0))(params, x.to(dtype=torch.float64), target)

    # def _r_principle_curvature_1(self, params, x):
    #     k_m = self._r_mean_curvature(params, x, 0)
    #     k_g = self._r_gauss_curvature(params, x, 0)
    #     k_1 = k_m + torch.sqrt(k_m.pow(2) - k_g)
    #     return k_1

    # def r_principle_curvature_1(self, params, x):
    #     return vmap(self._r_principle_curvature_1, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    # def _grad_theta_r_principle_curvature_1(self, params, x):
    #     return jacrev(self._r_principle_curvature_1, argnums=0)(params, x.to(dtype=torch.float64))

    # def grad_theta_r_principle_curvature_1(self, params, x):
    #     return vmap(self._grad_theta_r_principle_curvature_1, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    # def _r_principle_curvature_2(self, params, x):
    #     k_m = self._r_mean_curvature(params, x)
    #     k_g = self._r_gauss_curvature(params, x, 0)
    #     k_2 = k_m - torch.sqrt(k_m.pow(2) - k_g)
    #     return k_2

    # def r_principle_curvature_2(self, params, x):
    #     return vmap(self._r_principle_curvature_2, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    # def _grad_theta_r_principle_curvature_2(self, params, x):
    #     return jacrev(self._r_principle_curvature_2, argnums=0)(params, x.to(dtype=torch.float64))

    # def grad_theta_r_principle_curvature_2(self, params, x):
    #     return vmap(self._grad_theta_r_principle_curvature_2, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

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
