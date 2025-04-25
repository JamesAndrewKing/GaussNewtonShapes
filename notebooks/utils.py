"""Part of Arturs' isochrone code"""
'''Helper functions'''
import torch
from time import time

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def norm_sq(x):
    '''Square-norm of N vectors gather in tensor of shape [N,D]'''
    return x.square().sum(1)

def norm(x):
    '''Norm of vectors N gather in tensor of shape [N,D]'''
    return norm_sq(x).sqrt()

def normalize(dfdx):
    '''Normalize dfdx to have normals. Both input and output are shape [N,D]'''
    return dfdx/norm(dfdx)[:,None]

def grads_to_basis(dfdx, dfdO): ## TODO: this and lstsq are probably a key part of the library
    '''Compute basis for LSTSQ from the gradients
    dfdx: tensor wrt coordinates of shape [N,D]
    dfdO: tensor wrt (latent) parameters of shape [N,P]
    '''
    return torch.einsum('i,ip->ip',-1/norm(dfdx),dfdO)

def solve_lstsq(B, c, lmbd=2, verbose=False, method='lstsq'):
    '''Least-squares of form Bx=c
    B: basis, tensor of shape [N,P]
    c: rhs, tensor of shape [N] (target deformation projected onto normal)
    '''
    assert method in ["lstsq", "normalsolve"], "Available methods are lstsq and normalsolve"
    with torch.no_grad():
        ## Driver depending on condition number and device
        ## driver=gels  is fast, but sometimes instable for high condition numbers
        ## driver=gelsd is much more stable, but also much slower
        if verbose: t0 = time()

        if method=='lstsq':
            if lmbd:
                ## Tikhonov regularization
                B_tik = torch.vstack([B, lmbd*torch.eye(len(B.T), device=B.device)])
                c_tik = torch.hstack([c,    torch.zeros(len(B.T), device=B.device)])
                lstsq = torch.linalg.lstsq(B_tik, c_tik, driver='gels')
            else:
                lstsq = torch.linalg.lstsq(B, c, driver='gels')
            sol = lstsq.solution
        elif method=='normalsolve':
            '''
            Assemble the normal equation ourselves,
            add regularization onto diagonal
            and torch.linalg.solve the square system.
            Less accurate than lstsq, but less memory requirements.
            For large lmbd they converge to the same.
            '''
            BTB = B.T@B
            BTc = B.T@c
            BTB += lmbd*torch.eye(len(BTB))
            sol = torch.linalg.solve(BTB, BTc)

        if verbose:
            print(f"Solved lstsq in {(time()-t0):.4f} s")
            print(f"Largest sol entry: {sol.abs().max():.2e}")
            approx = B@sol
            mean_res = (approx - c).square().sum()/len(c)
            print(f"Mean residual: {mean_res:.2e}")

        return sol

def update_params(model, dO):
    '''
    Manually update the model parameters.
    model: torch.nn.Module to be updated
    dO: parameter update (premultiplied with the learning rate), tensor of correct shape [P]
    NOTE: modifies model.parameters in-place
    '''
    shapes = [param.shape for param in model.parameters()]
    with torch.no_grad():
        idx_stop = 0
        for param, shape in zip(model.parameters(), shapes):
            ## Pick the correct part in the vector and reshape
            idx_start = idx_stop
            idx_stop = idx_start + torch.numel(param)
            y_param = dO[idx_start:idx_stop].reshape(shape)
            ## Update the weights with a learning rate
            new_param = param + y_param
            param.copy_(new_param)

def get_mean_curvature(F, H):
    '''
    Mean-curvature in D-dimensions
    F: grad(f) gradients, shape [N,D]
    H: hess(f) Hessian, shape [N,D,D]

    https://u.math.biu.ac.il/~katzmik/goldman05.pdf
    For a shape implicitly defined by f<0:
    - div(F/|F|) = -(FHF^T - |F|^2 tr(H)) / 2*|F|^3
    In <=3D we can expand the formula, if we want to validate https://www.archives-ouvertes.fr/hal-01486547/document
    fx, fy, fz = F.T
    fxx, fxy, fxz, fyx, fyy, fyz, fzx, fzy, fzz = H.flatten(start_dim=1).T
    k = (fx*fx*(fyy+fzz) + fy*fy*(fxx+fzz) + fz*fz*(fxx+fyy) - 2*(fx*fy*fxy+fx*fz*fxz+fy*fz*fyz)) / (2*(fx*fx+fy*fy+fz*fz).pow(3/2))
    '''
    ## Quadratic form
    FHFT = torch.einsum('bi,bij,bj->b', F, H, F)
    ## Trace of Hessian
    trH = torch.einsum('bii->b', H)
    ## Norm of gradient
    N = F.square().sum(1).sqrt()
    ## Mean-curvature
    return -(FHFT - N.pow(2)*trH) / (2*N.pow(3))

def filter_bbox(pts, bbox):
    bbox = torch.tensor(bbox).T
    mask = torch.logical_and(bbox[0]<pts, pts<bbox[1]).all(1)
    return pts[mask]