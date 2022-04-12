import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        """ put all batch norm modules into evaluation mode
            this leads to the modules not using the current batch statistics,
            but instead using the overall mean and variance estimates collected during training

            why only freeze instances of batch norm -> other norms may be chosen in arguments
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape

        # create two coordinate grids with shape (batch, 2, H//8, W//8)
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape

        # mask dimensions: (N: batch, 1, 9: weights per upsampled pixel,
        # 8: number of upsampled pixels in height dir,
        # 8: number of upsampled pixels in width dir, image height, image width)
        mask = mask.view(N, 1, 9, 8, 8, H, W)

        # make weights for each upsampled pixel sum to one
        mask = torch.softmax(mask, dim=2)

        # copy 3x3 windows around each pixel, 
        # shape: (batch, 2, ht, wd) -> (batch, 2*3*3, number_of_blocks: ht*wd)
        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        # shape: (batch, 2*3*3, number_of_blocks: ht*wd) -> (batch, 2, 9, 1, 1, ht, wd)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        # do the multiplication of the 9 weights with the 9 pixels
        # (N, 1, 9, 8, 8, H, W) * (N, 2, 9, 1, 1, H, W) -> (N, 2, 9, 8, 8, H, W)
        # sum over the weighted pixel values for each upsampled pixel in the (8,8) window
        # shape: (N, 2, 9, 8, 8, H, W) -> (N, 2, 8, 8, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        
        # permute: (N, 2, 8, 8, H, W) -> (N, 2, H, 8, W, 8)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        # reshape (itegrate upsampled pixels into height/width dimension)
        # shape: (N, 2, H, 8, W, 8) -> (N, 2, 8*H, 8*W)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # map color values [0,255] -> [-1,1]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        # make image tensors contiguous (re-initialize tensor with memory format accomodating the tensors metadata)
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        # cast the per-8x8-pixel features to float32
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            # creates the correlation volume 
            # it is wrapped in an object with methods for querying using coordinate grid
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network (provides initial hidden state and context features)
        with autocast(enabled=self.args.mixed_precision):
            # the context network produces the initial hidden state (net) and the context features (inp) 
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # coords0 and coords1 are initialized to coordinate grids of shape (batch, 2, ht, wd)
        coords0, coords1 = self.initialize_flow(image1)

        # as: 
        # flow = coords1 - coords0; 
        # then adding the inital flow to coords1 causes: 
        # flow = coords1 + flow_init - coords0 = flow_init
        if flow_init is not None:
            coords1 = coords1 + flow_init

        # flow predictions at every stage are stored in a list
        flow_predictions = []

        # repeat optimization
        for itr in range(iters):

            # detaching coords1 from current computation graph
            # this makes sense because gradient should not be propagated on the route through coords1 to previous optimization stages
            # another reason would be because the flow calculated from coords1 is used for indexing 
            coords1 = coords1.detach()
            
            # call method of the correlation volume object
            # queries correlation values at all correlation volume levels
            # resulting shape: (batch, num_levels*dim*(2*r+1)*(2*r+1), h1, w1)
            corr = corr_fn(coords1) # index correlation volume

            # recompute flow from coords1
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                # update block receives inital hidden state, context features, correlation features and current flow estimate
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                # simple bilinear upsampling of the flow
                flow_up = upflow8(coords1 - coords0)
            else:
                # upsampling based on the upsampling weights produced 
                # by hidden state + mask convolutions in update module
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
