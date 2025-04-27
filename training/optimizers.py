import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from training.gram_factory import compute_loss, compute_residual, compute_JTJ, compute_JJT, compute_Jv
# from .gram_factory import compute_loss, compute_JJT


# This optimizer uses faster jvps, but the Woodbury trick is not implemented
class GaussNewton:
    def __init__(self, model, config, lr=0.1, do_line_search=False, line_search_steps=15):
        self.model = model
        self.params_dict = dict(model.named_parameters())
        self.params_list = list(model.parameters())
        self.lr = lr
        self.t = 0
        self.config = config
        self.do_line_search = do_line_search
        self.line_search_steps = line_search_steps
        self.loss = 1e5
        self.flat_update_direction = None
    
    @torch.no_grad()
    def zero_grad(self):
        for p in self.params_list:
            if p.grad is not None:
                p.grad.zero_()

    def apply_preconditioner_to_grads(self):
        """
        Solve the least squares problem: x = argmin ||A*x - grads||^2
        and assign the solution back into each parameter's .grad.
        """
        # 1. Flatten all gradients into a single vector.
        grads = [p.grad for p in self.params_list if p.grad is not None]
        flat_grads = parameters_to_vector(grads)
        N = flat_grads.numel()
        eps = self.config.get("regularization")
        A = compute_JJT(self.params_dict, self.config) + eps*torch.eye(N)

        # 2. Solve least squares: x = argmin_x ||A*x - grads||^2
        x, _, _, _ = torch.linalg.lstsq(A.double(), flat_grads.double(), driver="gels")
        x = x.to(flat_grads.dtype)
        self.flat_update_direction = x

        # 3. Unflatten x back into each parameter’s .grad
        vector_to_parameters(x, grads)
    
    @torch.no_grad()
    def step(self):
        """
        Perform a single Gauss-Newton update step on all parameters:
        """
        self.t += 1

        self.apply_preconditioner_to_grads()

        # if not self.do_line_search:
        #     for param in self.params_list:
        #         if param.grad is None:
        #             continue
        #         param.data -= self.lr * param.grad
        # If line search is enabled
        # TODO: do line search like in GaussNewtonNew, seems to work better
        if self.do_line_search:
            current_loss = compute_loss(self.params_dict, self.config)
            best_loss = current_loss
            best_lr = self.lr
            original_params = [param.clone() for param in self.params_list]
            best_params = original_params
            lrs = torch.logspace(0, -3, steps=self.line_search_steps)
        
            for lr in lrs:
                for param in self.params_list:
                    if param.grad is not None:
                        # param.data -= lr * param.grad
                        param.add_(param.grad, alpha=-lr)
                        
        
                params_dict = dict(zip(self.params_dict.keys(), self.params_list))
                new_loss = compute_loss(params_dict, self.config)
        
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_lr = lr
                    best_params = [param.clone() for param in self.params_list]
                else:
                    for i, param in enumerate(original_params):
                        self.params_list[i].data.copy_(param.data)
        
            self.lr = best_lr
            for i, param in enumerate(best_params):
                self.params_list[i].data.copy_(param.data)
        else:
            for param in self.params_list:
                if param.grad is None:
                    continue
                # param.data -= self.lr * param.grad
                param.add_(param.grad, alpha=-self.lr)


        # if self.do_line_search:
        #     # Save original parameters
        #     theta_orig = parameters_to_vector(self.params_list)
        #     best_loss = float('inf')
        #     best_theta = theta_orig
        #     best_lr = self.lr
        
        #     lrs = torch.logspace(0, -3, steps=self.line_search_steps, dtype=theta_orig.dtype)
        #     param_keys = list(self.params_dict.keys())
        #     for lr in lrs:
        #         theta_try = theta_orig - lr * self.flat_update_direction
        #         vector_to_parameters(theta_try, self.params_list)
        
        #         params_dict = dict(zip(param_keys, self.params_list))
        #         loss = compute_loss(params_dict, self.config)
        
        #         if loss < best_loss:
        #             best_loss = loss
        #             best_theta = theta_try
        #             best_lr = lr
        
        #     # Restore best found parameters
        #     vector_to_parameters(best_theta, self.params_list)
        #     self.lr = best_lr
        
        # else:
        #     # Simple step
        #     for param in self.params_list:
        #         if param.grad is not None:
        #             # param.data -= self.lr * param.grad
        #             param.add_(param.grad, alpha=-self.lr)


# This optimizer is slightly slower, but can handle big models using the Woodbury trick
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
        A = compute_JJT(self.params_dict, self.config) + eps * torch.eye(grad.shape[0], dtype=grad.dtype)
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
            
        
