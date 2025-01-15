import math
import einops
import torch

from models.net_w_partials import NetWithPartials
from models.point_wrapper import PointWrapper


def get_grid_starting_pts(n_shapes, x_grid, grid_dist):
    '''
    Create grid once at the beginning.
    Translate the grid by a random offset.
    '''
    nx = x_grid.shape[1]
    
    ## Translate the grid by a random offset
    xc_offset = torch.rand((n_shapes, nx)) * grid_dist  # bz nx

    # x_grid: [n_points nx]
    x = x_grid.unsqueeze(0) + xc_offset.unsqueeze(1)  # bz n_points nx

    ## Translate each point by a random offset
    x += torch.randn(x_grid.shape) * grid_dist / 3

    return PointWrapper.create_from_equal_bx(x)
    

def find_boundary_points_numerically_with_binsearch(x_grid, x_grid_dist, level_set, netp: NetWithPartials, z, n_steps=5, resolution=None):
    
    nx = len(x_grid_dist)
    
    p_grid = get_grid_starting_pts(len(z), x_grid, x_grid_dist)
    y_grid = netp.grouped_no_grad_fwd('vf', p_grid.data, p_grid.z_in(z)).squeeze(1)

    if resolution is None:
        # Infer resolution for a uniform grid
        n_points_root = int(round(math.pow(x_grid.shape[0], 1. / nx)))
        assert n_points_root**nx == x_grid.shape[0], "y_grid shape does not match nx dimensions."
        resolution = [n_points_root] * nx

    # Reshape the grid
    y_grid_reshaped = einops.rearrange(y_grid, 
                                    f'(b {" ".join([f"x{i}" for i in range(nx)])}) -> b {" ".join([f"x{i}" for i in range(nx)])}', 
                                    b=len(z), **{f'x{i}': resolution[i] for i in range(nx)})

    # Define masks for points above and below the level set
    above_level_set = y_grid_reshaped > level_set
    below_level_set = y_grid_reshaped < level_set
    boundary_inside_mask = torch.zeros_like(above_level_set, dtype=torch.bool)
    
    # create valid mask for boundary points of the domain 
    valid_mask = torch.ones_like(y_grid_reshaped, dtype=torch.bool)
    for d in range(1, valid_mask.dim(), 1):
        slice_indices = [slice(None)] * valid_mask.dim()
        slice_indices[d] = 0
        valid_mask[tuple(slice_indices)] = False
        slice_indices[d] = -1
        valid_mask[tuple(slice_indices)] = False

    # Define shifts
    shifts = [tuple([0] + [1 if i == d else 0 for i in range(nx)]) for d in range(nx)] + \
            [tuple([0] + [-1 if i == d else 0 for i in range(nx)]) for d in range(nx)]
    
    # Find points inside the boundary 
    for shift in shifts:
        shifted_below_level_set = torch.roll(below_level_set, shifts=shift, dims=tuple(range(nx+1)))
        boundary_inside_mask = boundary_inside_mask | (above_level_set & shifted_below_level_set & valid_mask)

    # Points inside the boundary mask
    p_inside = p_grid.select_w_mask(boundary_inside_mask.flatten())
    if len(p_inside) == 0:
        return False, (None, None)

    # Initialize outside points
    x_outside = p_inside.data.clone()
    z_in = p_inside.z_in(z)
    # Perturbation to find initial outside points
    for d in range(nx):
        perturb = torch.zeros_like(x_outside)
        perturb[:, d] = x_grid_dist[d] * 1.5
        outside_perturb_plus = p_inside.data + perturb
        outside_perturb_minus = p_inside.data - perturb

        # Evaluate level set to find which perturbations are outside
        level_plus = netp.grouped_no_grad_fwd('vf', outside_perturb_plus, z_in).squeeze(1)
        level_minus = netp.grouped_no_grad_fwd('vf', outside_perturb_minus, z_in).squeeze(1)

        # Update outside points
        plus_mask = level_plus < level_set
        x_outside[plus_mask] = outside_perturb_plus[plus_mask]
        minus_mask = level_minus < level_set
        x_outside[minus_mask] = outside_perturb_minus[minus_mask]

    p_outside = PointWrapper(x_outside, p_inside._map)

    # Binary search refinement
    high = p_inside.data
    low = p_outside.data
    z_in = p_inside.z_in(z)
    for i in range(n_steps):
        mid = (low + high) / 2
        level = netp.grouped_no_grad_fwd('vf', mid, z_in).squeeze(1)
        # print(f'i: {i} level.mean(): {level.mean()}, level.std(): {level.std()}')
        above_level_set = level > level_set
        low = torch.where(above_level_set.unsqueeze(1), low, mid)
        high = torch.where(above_level_set.unsqueeze(1), mid, high)

    p_res = PointWrapper(data=(low + high) / 2, map=p_inside._map)
    y_sel = netp.grouped_no_grad_fwd('vf', p_res.data, p_res.z_in(z)).squeeze(1)
    # print(f'final level.mean(): {y_sel.mean()}, level.std(): {y_sel.std()}')
    return True, (p_res, y_sel)
