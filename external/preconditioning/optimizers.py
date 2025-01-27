import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from external.preconditioning.gram_factory import compute_gram_matrix

class HandAdam:
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """
        params: iterable of torch.Parameters (e.g. model.parameters())
        lr: learning rate
        betas: (beta1, beta2) coefficients
        eps: small number for numerical stability
        """
        self.params_dict = dict(model.named_parameters())
        self.params = list(model.parameters())
        self.lr = lr
        self.betas = betas
        self.eps = eps

        self.t = 0  # keep track of update steps
        # For each param, maintain its own m (first moment) and v (second moment)
        self.m = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params]
        self.v = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params]

    def zero_grad(self):
        """Set gradients of all optimized parameters to zero."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        """
        Perform a single Adam update step on all parameters.
        """
        self.t += 1
        beta1, beta2 = self.betas

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            g = param.grad  # gradient
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g      # update first moment
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (g*g)  # update second moment

            # Correct bias in moments
            m_hat = self.m[i] / (1 - beta1**self.t)
            v_hat = self.v[i] / (1 - beta2**self.t)

            # Update parameter
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

class HandNGD:
    def __init__(self, model, lr, config):
        """
        Implements basic gradient descent.
        
        Args:
            params: iterable of torch.Parameters (e.g., model.parameters())
            lr    : learning rate
        """
        self.model = model
        self.params_dict = dict(model.named_parameters())
        self.params = list(model.parameters())  # store references to model parameters
        self.lr = lr
        self.t = 0
        self.config = config
        self.loss = 1e6
        self.old_loss = 1e6
        self.old_params = self.params

    def zero_grad(self):
        """Set gradients of all optimized parameters to zero."""
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
        flat_grads = parameters_to_vector(grads)  # shape: (N,)
        N = flat_grads.numel()
        eps = self.config.get("regularization")
        A = compute_gram_matrix(self.model, self.config) + eps*torch.eye(N)

        # 2. Solve least squares: x = argmin_x ||A*x - grads||^2
        x, _, _, _ = torch.linalg.lstsq(A.double(), flat_grads.double(), driver="gels")

        # 3. Unflatten x back into each parameter’s .grad
        vector_to_parameters(x, grads)

    def step(self):
        """
        Perform a single precondtioned gradient descent update step on all parameters:
        """
        self.t += 1

        if self.loss <= 2 * self.old_loss:

            self.old_params = self.params
            self.old_loss = self.loss

            self.apply_preconditioner_to_grads()
    
            for param in self.params:
                if param.grad is None:
                    continue
    
                param.data -= self.lr * param.grad
        else:
            print("Not updating parameters because of large loss metric increase")
            self.params = self.old_params