import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from external.preconditioning.gram_factory import compute_gram_matrix

class GaussNewton:
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

        self.apply_preconditioner_to_grads()

        for param in self.params:
            if param.grad is None:
                continue

            param.data -= self.lr * param.grad
