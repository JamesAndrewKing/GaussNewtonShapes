import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .gram_factory import compute_gram_matrix, compute_residual, compute_JTJ, compute_JJT, compute_Jv

class GaussNewton:
    def __init__(self, model, lr, config):
        self.model = model
        self.params_dict = dict(model.named_parameters())
        self.params = list(model.parameters())
        self.lr = lr
        self.t = 0
        self.config = config

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def apply_preconditioner_to_grads(self):
        """
        Solve the least squares problem: x = argmin ||A*x - grads||^2
        and assign the solution back into each parameter's .grad.
        """
        # 1. Flatten all gradients into a single vector.
        grads = [p.grad for p in self.params if p.grad is not None]
        flat_grads = parameters_to_vector(grads)
        N = flat_grads.numel()
        eps = self.config.get("regularization")
        A = compute_gram_matrix(self.model, self.config) + eps*torch.eye(N)

        # 2. Solve least squares: x = argmin_x ||A*x - grads||^2
        x, _, _, _ = torch.linalg.lstsq(A.double(), flat_grads.double(), driver="gels")

        # 3. Unflatten x back into each parameter’s .grad
        vector_to_parameters(x, grads)

    def step(self):
        """
        Perform a single Gauss-Newton update step on all parameters:
        """
        self.t += 1

        self.apply_preconditioner_to_grads()

        for param in self.params:
            if param.grad is None:
                continue

            param.data -= self.lr * param.grad

class GaussNewtonNew:
    def __init__(self, model, config, lr=0.1, do_line_search=True, line_search_steps=15, do_woodbury=False):
        self.model = model
        self.params_dict = dict(model.named_parameters())
        self.params = list(model.parameters())
        self.lr = lr
        self.t = 0
        self.config = config
        self.do_line_search = do_line_search
        self.line_search_steps = line_search_steps
        self.do_woodbury = do_woodbury
        self.loss = 1e5

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def calculate_update(self):
        r = compute_residual(self.model, self.config)
        grad = compute_Jv(self.model, self.config, r)
        eps = self.config.get("regularization", 0.0)
        A = compute_JJT(self.model, self.config) + eps * torch.eye(grad.shape[0], dtype=grad.dtype)
        x, _, _, _ = torch.linalg.lstsq(A.double(), grad.double(), driver="gels")
        return x

    def calculate_update_woodbury(self):
        r = compute_residual(self.model, self.config)
        eps = self.config.get("regularization", 0.0)
        A = compute_JTJ(self.model, self.config) + eps * torch.eye(r.shape[0], dtype=r.dtype, device=r.device)
        x, _, _, _ = torch.linalg.lstsq(A.double(), r.double(), driver="gels")
        return compute_Jv(self.model, self.config, x)

    @torch.no_grad()
    def step(self):
        self.t += 1

        if not self.do_woodbury:
            update = self.calculate_update()
        else:
            update = self.calculate_update_woodbury()
    
        theta_orig = parameters_to_vector(self.params)
    
        if self.do_line_search:
            best_loss = float('inf')
            best_theta = theta_orig
            best_lr = self.lr
    
            lrs = torch.logspace(0, -3, steps=self.line_search_steps, dtype=theta_orig.dtype)
    
            for lr in lrs:
                theta_try = theta_orig - lr * update
                vector_to_parameters(theta_try, self.params)
    
                residual = compute_residual(self.model, self.config)
                loss = torch.sum(residual.square()).item()
    
                if loss < best_loss:
                    best_loss = loss
                    self.loss = best_loss
                    best_theta = theta_try
                    best_lr = lr
    
            vector_to_parameters(best_theta, self.params)
            self.lr = best_lr
    
        else:
            # Single update step without line search
            theta_new = theta_orig - self.lr * update
            vector_to_parameters(theta_new, self.params)


# class GaussNewtonWoodburyBig:
#     def __init__(self, model, lr, config):
#         self.model = model
#         self.params_dict = dict(model.named_parameters())
#         self.params = list(model.parameters())
#         self.lr = lr
#         self.t = 0
#         self.config = config
#         self.do_line_search = True
#         self.line_search_steps = 10

#     def zero_grad(self):
#         for p in self.params:
#             if p.grad is not None:
#                 p.grad.zero_()

#     def calculate_update(self):
#         r = compute_residual(self.model, self.config)
#         eps = self.config.get("regularization", 0.0)
#         A = compute_JTJ(self.model, self.config) + eps * torch.eye(r.shape[0], dtype=r.dtype, device=r.device)
#         x, _, _, _ = torch.linalg.lstsq(A.double(), r.double(), driver="gels")
#         return compute_Jv(self.model, self.config, x)

#     @torch.no_grad()
#     def step(self):
#         self.t += 1
#         update = self.calculate_update()
    
#         theta_orig = parameters_to_vector(self.params)
    
#         if self.do_line_search:
#             best_loss = float('inf')
#             best_theta = theta_orig
#             best_lr = self.lr
    
#             lrs = torch.logspace(0, -3, steps=self.line_search_steps, dtype=theta_orig.dtype, device=theta_orig.device)
    
#             for lr in lrs:
#                 theta_try = theta_orig - lr * update
#                 vector_to_parameters(theta_try, self.params)
    
#                 residual = compute_residual(self.model, self.config)
#                 loss = torch.sum(residual.square()).item()
    
#                 if loss < best_loss:
#                     best_loss = loss
#                     best_theta = theta_try
#                     best_lr = lr
    
#             vector_to_parameters(best_theta, self.params)
#             self.lr = best_lr
    
#         else:
#             # Single update step without line search
#             theta_new = theta_orig - self.lr * update
#             vector_to_parameters(theta_new, self.params)
            
        
