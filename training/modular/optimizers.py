import torch
from training.modular.residuals import compute_loss
from training.modular.gram_factory import compute_JTJ
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class GaussNewton:
    def __init__(self, model, res_terms, lr=0.1, regularization=1e-6, do_line_search=False, line_search_steps=15):
        self.params_dict = dict(model.named_parameters())
        self.params_list = list(model.parameters())
        self.res_terms = res_terms
        self.lr = lr
        self.regularization = regularization
        self.do_line_search = do_line_search
        self.line_search_steps = line_search_steps
        self.t = 0
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
        eps = self.regularization
        A = compute_JTJ(self.params_dict, self.res_terms) + eps*torch.eye(N)

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
        if self.do_line_search:
            current_loss = compute_loss(self.params_dict, self.res_terms)
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
                new_loss = compute_loss(params_dict, self.res_terms)
        
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