import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        """calculate per-pixel features with motion components as inputs (flow estimate, correlation features)

        Args:
            args (Namespace): Arguments passed to RAFT object creation
        """
        super(BasicMotionEncoder, self).__init__()

        # correlation feature size (with correlation value dimension set to one - left out here)
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        
        # convolution layers for correlation features (convolution over pixels, channels correspond to correlation feature indices)
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)  # keeping ht/wd identical due to padding
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)         # keeping ht/wd identical
        
        # convolution layers for flow estimate features (convolution over pixels, channels are x/y coordinate values)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)           # keeping ht/wd identical
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)          # keeping ht/wd identical
        
        # convolution layer combining flow estimate features and correlation features
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)      # keeping ht/wd identical

    def forward(self, flow, corr):
        """receives flow estimate and correlation features, calculates motion features

        Args:
            flow (torch.Tensor): current flow estimate (batch, 2, ht, wd)
            corr (torch.Tensor): current correlation features (batch, num_corr_levels*(2*r+1)**2, ht, wd)

        Returns:
            torch.Tensor: _description_
        """

        # calculate correlation features features 
        # (batch, num_corr_levels*(2*r+1)**2, ht, wd) -> (batch, 192, ht, wd)
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        
        # calculate flow estimate features
        # (batch, 2, ht, wd) -> (batch, 64, ht, wd)
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        # concatenation
        # (batch, 64, ht, wd), (batch, 192, ht, wd) -> (batch, 192+64, ht, wd)
        cor_flo = torch.cat([cor, flo], dim=1)
        
        # flow/correlation features combination layer
        # (batch, 192+64, ht, wd) -> (batch, 128-2, ht, wd)
        out = F.relu(self.conv(cor_flo))
        
        # concatenation
        # (batch, 128-2, ht, wd), (batch, 2, ht, wd) -> (batch, 128, ht, wd)
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        """_summary_

        Args:
            net (torch.Tensor): previous hidden state
            inp (torch.Tensor): context features
            corr (torch.Tensor): correlation features
            flow (torch.Tensor): current flow estimate
            upsample (bool, optional): Remains unused. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: new hidden state, 
        """
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balance gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow



