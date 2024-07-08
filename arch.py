import torchvision
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t


def trans_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))

    else:
        k_slices = []
        b_slices = []

        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups

        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) * k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]

            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append(
                (k2_slice * b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))

        k = torch.cat(k_slices, dim=0)
        b_hat = torch.cat(b_slices)

    return k, b_hat + b2


def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == "gelu":
        layer = nn.GELU()
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


class ResBlock(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.
    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|
    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(ResBlock, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, int(ratio * n_feats), 1, 1, 0)
        self.fea_conv = nn.Conv2d(int(ratio * n_feats), int(ratio * n_feats), 3, 1, 0)
        self.reduce_conv = nn.Conv2d(int(ratio * n_feats), n_feats, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out

        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return out


class RepResBlock(nn.Module):
    def __init__(self, n_feats):
        super(RepResBlock, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)

        return out


class ECALayer(nn.Module):
    """
    Constructs an efficient channel attention layer.
    """

    def __init__(self, channels, gamma=1, b=1):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SimplifiedNAFBlock(nn.Module):
    def __init__(self, in_c, exp, act, eca_gamma, layernorm, residual) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = ResBlock(in_c, exp)

        # activation
        self.act = act
        # channel attention
        if eca_gamma > 0:
            self.ca = ECALayer(in_c, gamma=eca_gamma)
        else:
            self.ca = None
        # layernorm
        if layernorm:
            self.norm = LayerNorm2d(in_c)
        else:
            self.norm = None

    def forward(self, x):
        res = x.clone()

        if self.norm is not None:
            x = self.norm(x)

        x = self.conv1(x)

        if self.act is not None:
            x = self.act(x)

        if self.ca is not None:
            x = self.ca(x) * x

        if self.residual:
            x += res
        else:
            x = x
        return x


class SimplifiedRepNAFBlock(nn.Module):
    def __init__(self, in_c, exp, act, eca_gamma, layernorm, residual) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = RepResBlock(in_c)

        # activation
        self.act = act

        # channel attention
        if eca_gamma > 0:
            self.ca = ECALayer(in_c, gamma=eca_gamma)
        else:
            self.ca = None

        # LayerNorm
        if layernorm:
            self.norm = LayerNorm2d(in_c)
        else:
            self.norm = None

    def forward(self, x):
        res = x.clone()

        if self.norm is not None:
            x = self.norm(x)

        x = self.conv1(x)

        if self.act is not None:
            x = self.act(x)

        if self.ca is not None:
            x = self.ca(x) * x

        if self.residual:
            x += res
        else:
            x = x
        return x


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class RT4KSR_Rep(nn.Module):
    def __init__(self,
                 num_channels=3,
                 num_feats=24,
                 num_blocks=4,
                 upscale=3,
                 act=nn.GELU(),
                 eca_gamma=0,
                 is_train=True,
                 forget=False,
                 layernorm=True,
                 residual=False) -> None:
        super().__init__()
        self.forget = forget
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)

        self.down = nn.PixelUnshuffle(2)
        self.up = nn.PixelShuffle(2)
        self.head = nn.Sequential(nn.Conv2d(num_channels * (2 ** 2), num_feats, 3, padding=1))

        hfb = []
        if is_train:
            hfb.append(ResBlock(num_feats, ratio=2))
        else:
            hfb.append((RepResBlock(num_feats)))
        hfb.append(act)
        self.hfb = nn.Sequential(*hfb)

        body = []
        for i in range(num_blocks):
            if is_train:
                body.append(SimplifiedNAFBlock(in_c=num_feats, act=act, exp=2, eca_gamma=eca_gamma, layernorm=layernorm,
                                               residual=residual))
            else:
                body.append(
                    SimplifiedRepNAFBlock(in_c=num_feats, act=act, exp=2, eca_gamma=eca_gamma, layernorm=layernorm,
                                          residual=residual))

        self.body = nn.Sequential(*body)

        tail = [LayerNorm2d(num_feats)]
        if is_train:
            tail.append(ResBlock(num_feats, ratio=2))
        else:
            tail.append(RepResBlock(num_feats))
        self.tail = nn.Sequential(*tail)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, num_channels * ((2 * upscale) ** 2), 3, padding=1),
            nn.PixelShuffle(upscale * 2)
        )

    def forward(self, x):
        # stage 1
        hf = x - self.gaussian(x)

        # unshuffle to save computation
        x_unsh = self.down(x)
        hf_unsh = self.down(hf)

        shallow_feats_hf = self.head(hf_unsh)
        shallow_feats_lr = self.head(x_unsh)

        # stage 2
        deep_feats = self.body(shallow_feats_lr)
        hf_feats = self.hfb(shallow_feats_hf)

        # stage 3
        if self.forget:
            deep_feats = self.tail(self.gamma * deep_feats + hf_feats)
        else:
            deep_feats = self.tail(deep_feats)

        out = self.upsample(deep_feats)
        return out
