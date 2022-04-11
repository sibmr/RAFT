import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates

    Args:
        img (torch.Tensor): correlation volume of shape (batch*h1*w1, dim, h2, w2)
        coords (torch.Tensor): coordinates to sample for each pixel of shape (batch*h1*w1, 2*r+1, 2*r+1, 2)
        mode (str, optional): Not accessed by program currently - could be passed to grid_sample. Defaults to 'bilinear'.
        mask (bool, optional): Whether to return the mask being 1.0 for pixels outside of grid,
                                same shape as return value. Defaults to False.

    Returns:
        torch.Tensor: sampled correlation values for the specified coordinates, shape (batch*h1*w1, dim, 2*r+1, 2*r+1)
                        correlation values sampled around the current estimated image2 location for each image1 pixel
    """
    
    # H = h2, W = w2
    H, W = img.shape[-2:]

    # split coords into x/y-values:
    # (batch*h1*w1, 2*r+1, 2*r+1, 2) -> (batch*h1*w1, 2*r+1, 2*r+1, 1), (batch*h1*w1, 2*r+1, 2*r+1, 1)
    xgrid, ygrid = coords.split([1,1], dim=-1)
    
    # map grid coordinate ranges: [0,W-1] -> [-1, 1] ; [0,H-1] -> [-1, 1]
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    # merge x/y-coords after range transformation:
    # (batch*h1*w1, 2*r+1, 2*r+1, 1), (batch*h1*w1, 2*r+1, 2*r+1, 1) -> (batch*h1*w1, 2*r+1, 2*r+1, 2)
    grid = torch.cat([xgrid, ygrid], dim=-1)

    # sampled correlation volume points of shape (batch*h1*w1, dim, 2*r+1, 2*r+1)
    img = F.grid_sample(img, grid, align_corners=True)

    # create mask with 1.0 for coordinates outside the correlation volume, otherwise 0.0
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    """ creates a tensor which contains the coordinates of previous dimensions

    Args:
        batch (int): batch size
        ht (int): image height
        wd (int): image width
        device (int): device for computation

    Returns:
        torch.Tensor: _description_
    """

    # resulting coords tensor looks like this:
    # tensor(  [[[0., 1., 2., 3., 4.],
    #           [0., 1., 2., 3., 4.],
    #           [0., 1., 2., 3., 4.],
    #           [0., 1., 2., 3., 4.]],

    #          [[0., 0., 0., 0., 0.],
    #           [1., 1., 1., 1., 1.],
    #           [2., 2., 2., 2., 2.],
    #           [3., 3., 3., 3., 3.]]])


    # first step: create meshgrid that is separate for each dimension: (ht,wd), (ht,wd)
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    # second step: create meshgrid that is reversed and merged with shape: (2,ht,wd)
    # reversal: (coords[0][i][j],coords[1][i][j]) = (j,i)
    coords = torch.stack(coords[::-1], dim=0).float()
    # add a batch dimension using [None] (shape (2, ht, wd) -> (1, 2, ht, wd))
    # and then clone the coords along the batch dimension: shape (1, 2, ht, wd) -> (batch, 2, ht, wd)
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
