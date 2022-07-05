import torch
from einops import rearrange
from torch import nn
import math
import numpy as np


class Residual(nn.Module):
    def __init__(self, fn):
        """ Residual layer. """
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """ Layer normalization. """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        """ FeedForward layer. """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class SepConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        """ Separable 2D convolution.

            Parameters
            ----------
            in_channels: int
                Number of input channels.
            out_channels: int
                Number of output channels.
            kernel_size: int
                Kernel size.
            stride: int
                Stride. (Default: 1)
            padding: int
                Padding. (Default: 0)
            dilation: int
                (Default: 1)
        """
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class ConvAttention(nn.Module):
    def __init__(self, heads=8, base_dim=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, attn_drop=0.,
                 proj_drop=0., apply_transform=True, transform_scale=False):
        """ Attention with convolutional projections.

            Parameters
            ----------
            heads: int
                Number of attention heads. (Default: 8)
            base_dim: int
                Dimension of each attention head. (Default: 64)
            kernel_size: int
                Kernel size for conv projection. (Default: 3)
            q_stride: int
                Stride for Q projection. (Default: 1)
            k_stride: int
                Stride for K projection. (Default: 1)
            v_stride: int
                Stride for V projection. (Default: 1)
            attn_drop: int
                Optional dropout attention map to prevent overfitting. (Default: 0)
            proj_drop: int
                Optional dropout for linear layer to prevent overfitting. (Default: 0)
            apply_transform: bool
                Apply re-attention. (Default: True)
            transform_scale: bool
                Re-attention scale. (Default: False)
        """
        super().__init__()
        self.heads = heads
        dim = heads
        self.base_dim = base_dim
        inner_dim = base_dim * heads
        self.scale = dim ** -0.5
        pad = (kernel_size - q_stride) // 2

        self.apply_transform = apply_transform
        self.to_q = SepConv2d(dim, dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, dim, kernel_size, v_stride, pad)
        self.to_out = nn.Linear(inner_dim, inner_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        if apply_transform:
            self.reatten_matrix = nn.Conv2d(self.heads, self.heads, 1, 1)
            self.var_norm = nn.BatchNorm2d(self.heads)
            self.reatten_scale = self.scale if transform_scale else 1.0

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        x = rearrange(x, 'b n (h d)-> b h n d', h=h)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, base_dim, num_heads, mlp_ratio, attn_drop=0., proj_drop=0.):
        """ A Transformer block.

            Parameters
            ----------
            dim: int
                Dimension of token embedding.
            base_dim: int
                Dimension of each attention head.
            num_heads: int
                Number of attention heads.
            mlp_ratio: int
                MLP expansion ratio.
        """
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList([])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.layers.append(nn.ModuleList([
            Residual(PreNorm(dim,
                             ConvAttention(heads=num_heads, base_dim=base_dim, kernel_size=3, q_stride=1, k_stride=1,
                                           v_stride=1, attn_drop=attn_drop, proj_drop=proj_drop, apply_transform=False,
                                           transform_scale=False))),
            Residual(PreNorm(dim, FeedForward(dim, mlp_hidden_dim)))
        ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio, attn_drop=0., proj_drop=0.):
        """ Stage level Transformer.

            Parameters
            ----------
            base_dim: int
                Attention head dimension.
            depth: int
                Number of transformer blocks.
            heads: int
                Number of attention heads.
            mlp_ratio: int
                MLP expansion ratio.
        """
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList([])
        embed_dim = base_dim * heads
        for _ in range(depth):
            self.blocks.append(TransformerBlock(
                dim=embed_dim,
                base_dim=base_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop
            ))

    def forward(self, x, cls_tokens):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')
        token_length = cls_tokens.shape[1]
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x, cls_tokens


class ConvHeadPooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        """ The class functions is that image channel is doubled, the width and height are halved, and the cls_token dimension is doubled

            Parameters
            ----------
            in_feature: int
                Number of input image channels.
            out_feature: int
                Number of output image channels.
            stride: int
                Stride. (Default: 2)
        """
        super(ConvHeadPooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):
        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        """ 2D convolutional embedding.

            Parameters
            ----------
            in_channels: int
                Number of input image channels.
            out_channels: int
                Number of output image channels.
            patch_size: int
                Patch size.
            stride: int
                Strides.
            padding: int
                Padding.
        """
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class Network(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads, mlp_ratio, num_classes=2, in_chans=512,
                 attn_drop=0., proj_drop=0.):
        """ The overall model network.

            Parameters
            ----------
            image_size: int
                Input image size.
            patch_size: int
                Patch size.
            stride: int
                Strides.
            base_dims: list
                Dimensions of each attention head for each stage of the transformer.
            depth: list
                Number of TransformerBlocks for each stage of transformer.
            heads: list
                Number of attention heads for each stage of the transformer.
            mlp_ratio: int
                MLP expansion ratio.
            num_classes: int
                Number of categories. (Default: 2)
            in_chans: int
                The number of input channels of the image after conv feature extraction. (Default: 512)
            attn_drop: float
                Optional attention map dropout rate. (Default: 0)
            drop_path: float
                Optional attention map dropout rate. (Default: 0)
        """

        super(Network, self).__init__()
        padding = 1
        width = math.floor(
            (image_size / 32 + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.patch_size = patch_size
        self.pos_embed = self.get_sinusoid_encoding(n_position=50, d_hid=512)
        self.patch_embed = ConvEmbedding(in_chans, base_dims[0] * heads[0], patch_size, stride, padding)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, base_dims[0] * heads[0]),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=proj_drop)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop
                            )
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    ConvHeadPooling(base_dims[stage] * heads[stage],
                                    base_dims[stage + 1] * heads[stage + 1],
                                    stride=2
                                    )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.features(x)
        x = self.patch_embed(x)
        h_p = x.shape[2]
        w_p = x.shape[3]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        token_length = cls_tokens.shape[1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h_p, w=w_p)
        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)
        cls_tokens = self.norm(cls_tokens)
        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token

    def get_sinusoid_encoding(self, n_position, d_hid):
        ''' Sinusoid position encoding table
            Parameters
                ----------
                n_position: int
                 the number of patch tokens
                d_hid: int
                 the dimension of each patch token
        '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
