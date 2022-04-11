import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # correlation is computed on intialization -> new CorrBlock for every batch
        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        # the -dim- value probably specifies the dimension of the correlation
        # for the paper the correlation is scalar, meaning: dim==1
        batch, h1, w1, dim, h2, w2 = corr.shape
        
        # pooling across h2, w2 -> the batch dimension contains the pixels from all batches: batch*h1*w1
        # this means we pool over the correlation values for each pixel in image1 separately to all pixels in image2  
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        # save correlation pyramid levels to list
        self.corr_pyramid.append(corr)

        for i in range(self.num_levels-1):
            # do the actual average pooling operation to create a new pyramid level (kernel_size=2, stride=2)
            # no overlap between kernel applications with kernel size 2, stride 2
            corr = F.avg_pool2d(corr, 2, stride=2)

            # save correlation pyramid levels to list
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        """ query the correlation volume for a flow estimate

        Args:
            coords (torch.Tensor): pixel coordinates for image2, in relation to image1

        Returns:
            torch.Tensor: _description_
        """

        # shorthand for grid radius
        r = self.radius

        # permutation: (batch, 2, ht, wd) -> (batch, ht, wd, 2)
        coords = coords.permute(0, 2, 3, 1)

        # get batch, ht, wd sizes
        batch, h1, w1, _ = coords.shape

        # store features from pyramid levels to list
        out_pyramid = []

        for i in range(self.num_levels):
            # get correlation volume at the current level
            corr = self.corr_pyramid[i]

            # get local grid coordinates relative to pixel center
            # dx = dy = [-r, -r+1, ..., -1, 0, 1, ..., r-1, r]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)

            # meshgrid of local coordinates: reversed since dy first
            # fitting since coords is also reversed with y first
            # this has shape (2*r+1, 2*r+1, 2)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            # shape (batch, ht, wd, 2) -> (batch*ht*wd, 1, 1, 2)
            # diveded by 2**pyramid_level accounting for the grid step size at each level
            # due to pooling and reduction in image size
            # delta is not diveded -> larger window at higher pyramid_level
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            
            # reshape (2*r+1, 2*r+1, 2) -> (1, 2*r+1, 2*r+1, 2)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)

            # coords (batch*ht*wd, 2*r+1, 2*r+1, 2) = centroid (batch*ht*wd, 1, 1, 2) + delta (1, 2*r+1, 2*r+1, 2)
            # for each pixel in image1, there are (2*r+1)**2 coordinates of locations in image2
            coords_lvl = centroid_lvl + delta_lvl

            # the bilinear sampler maps (2*r+1)**2 correlation values to each pixel
            # by using bilinear interpolation on the current correlation pyramid level (pooled values)
            # shape: (batch*h1*w1, 2*r+1, 2*r+1)
            corr = bilinear_sampler(corr, coords_lvl)
            
            # separate dimensions
            # (batch*h1*w1, dim, 2*r+1, 2*r+1) -> (batch, h1, w1, dim*(2*r+1)*(2*r+1))
            corr = corr.view(batch, h1, w1, -1)
            
            out_pyramid.append(corr)

        # the correlation values of all pyramid levels are concatenated
        # resulting shape: (batch, h1, w1, num_levels*dim*(2*r+1)*(2*r+1))
        out = torch.cat(out_pyramid, dim=-1)

        # permutation:
        # (batch, h1, w1, num_levels*dim*(2*r+1)*(2*r+1)) -> (batch, num_levels*dim*(2*r+1)*(2*r+1), h1, w1)
        # also contiguous is used to copy the tensor with new memory layout according to shape
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        """ create the correlation volume
            shape: (batch, ht, wd, 1, ht, wd)
        

        Args:
            fmap1 (torch.Tensor): feature map of image 1 of shape (batch, fdim, ht, wd)
            fmap2 (torch.Tensor): feature map of image 2 of shape (batch, fdim, ht, wd)

        Returns:
            torch.Tensor: correlation volume of shape (batch, ht, wd, 1, ht, wd)
        """
        batch, dim, ht, wd = fmap1.shape

        # flatten feature maps along the height/width dimension
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        # shapes of matmul inputs: (batch, ht*wd, dim) ; (batch, dim, ht*wd)
        # shape of matmul output:  (batch, ht*wd, ht*wd)
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        
        # reshape to separate height and width dimensions
        corr = corr.view(batch, ht, wd, 1, ht, wd)

        # divide by sqrt(feature_dim) -> this reduces the magnitude of the features
        # TODO: why divide by this value exactly?
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
