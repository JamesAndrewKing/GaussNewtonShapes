import torch
from torch.func import vmap, jacrev, jacfwd, functional_call

class Residuals:
    def __init__(self, model):
        """Initialize with a model."""
        self.model = model

    def f(self, params, x):
        """Wrapper for functional_call."""
        return functional_call(self.model, params, x.to(dtype=torch.float64))

    def _d_theta_f(self, params, x):
        """Gradient of f with respect to parameters theta (non-vectorized, private)."""
        return jacrev(self.f, argnums=0)(params, x.to(dtype=torch.float64))

    def d_theta_f(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._d_theta_f, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _f_x(self, params, x):
        """Jacobian of f with respect to input x (non-vectorized, private)."""
        return jacrev(self.f, argnums=1)(params, x.to(dtype=torch.float64))

    # def vf_x(self, params, x):
    #     """Vectorized Jacobian of f with respect to input x."""
    #     return vmap(self._f_x, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _f_xx(self, params, x):
        """Hessian of f with respect to input x (non-vectorized, private)."""
        return jacfwd(self._f_x, argnums=1)(params, x.to(dtype=torch.float64))

    # def vf_xx(self, params, x):
    #     """Vectorized Hessian of f with respect to input x."""
    #     return vmap(self._f_xx, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _residual_laplacian(self, params, x):
        """Laplacian of f with respect to input x (non-vectorized, private)."""
        hessian = self._f_xx(params, x).squeeze(1)  # Compute the Hessian
        laplacian = torch.einsum('bii->b', hessian)  # Sum of the diagonal elements
        return laplacian

    def residual_laplacian(self, params, x):
        """Vectorized Laplacian of f."""
        return vmap(self._residual_laplacian, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _d_theta_residual_laplacian(self, params, x):
        """Laplacian of phi (non-vectorized, private)."""
        return jacrev(self._residual_laplacian, argnums=0)(params, x.to(dtype=torch.float64))

    def d_theta_residual_laplacian(self, params, x):
        """Vectorized Laplacian of phi."""
        return vmap(self._d_theta_residual_laplacian, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _residual_mean_curvature(self, params, x):
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

    def residual_mean_curvature(self, params, x):
        """Vectorized mean curvature of f."""
        return vmap(self._residual_mean_curvature, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))
    
    def _d_theta_f_mean_curvature(self, params, x):
        return jacrev(self._residual_mean_curvature, argnums=0)(params, x.to(dtype=torch.float64))

    def d_theta_residual_mean_curvature(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._d_theta_f_mean_curvature, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _residual_eikonal(self, params, x):
        """Eikonal residual of f with respect to input x (non-vectorized, private)."""
        return self._f_x(params, x).squeeze(1).square().sum(1).sqrt() - 1

    def residual_eikonal(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._residual_eikonal, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    def _d_theta_residual_eikonal(self, params, x):
        return jacrev(self.residual_eikonal, argnums=0)(params, x.to(dtype=torch.float64))

    def d_theta_residual_eikonal(self, params, x):
        """Vectorized gradient of f with respect to parameters theta."""
        return vmap(self._d_theta_residual_eikonal, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))

    # def _phi_x(self, params, x):
    #     """Gradient of phi with respect to input x (non-vectorized, private)."""
    #     return jacfwd(self._phi, argnums=1)(params, x.to(dtype=torch.float64))

    # def v_phi_x(self, params, x):
    #     """Vectorized gradient of phi with respect to input x."""
    #     return vmap(self._phi_x, in_dims=(None, 0), out_dims=(0))(params, x.to(dtype=torch.float64))
