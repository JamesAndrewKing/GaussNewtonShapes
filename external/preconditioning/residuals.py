import torch

def mean_curvature(model, pts_space):
    F = model.vf_x(model.params, pts_space).squeeze(1)
    H = model.vf_xx(model.params, pts_space).squeeze(1)
    ## Quadratic form
    FHFT = torch.einsum('bi,bij,bj->b', F, H, F)
    ## Trace of Hessian
    trH = torch.einsum('bii->b', H)
    ## Norm of gradient
    N = F.square().sum(1).sqrt()
    ## Mean-curvature
    mean_curvatures = -(FHFT - N.pow(2)*trH) / (2*N.pow(3))
    return mean_curvatures