import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from utils.checkpoint import load_checkpoint
from mmseg.utils import get_root_logger
from functools import partial
from utils.module import Attention,PreNorm, FeedForward, CrossAttention
import math

groups = 32

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features).cuda()
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features).cuda()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate, proj_dropout_rate, vis=False):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(proj_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = Attention(hidden_size=dim, num_heads=num_heads, attention_dropout_rate=attn_drop, proj_dropout_rate=drop, vis=vis)

        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Apply standard Multi-Head Self Attention
        attn_output, _ = self.attn(x)
        x = shortcut + self.drop_path(attn_output)

        # Feed Forward Network
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class PatchRecover(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=dim//2, num_groups=groups),
            nn.ReLU(inplace=True)
        )


    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.permute(0, 1, 2) # B ,C, L
        x = x.reshape(B, C, H, W)
        x = self.up(x) # B, C//2, H, W

        x = x.reshape(B, C//2, -1)
        x = x.permute(0, 2, 1)

        #x = Variable(torch.randn(B, H * 2, W * 2, C // 2))

        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x





class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

class MultiEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.maxPool = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        x = self.bn(x)
        x = self.maxPool(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x



class SwinTransformer(nn.Module):
    
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
            
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.drop_path_rate=drop_path_rate
        self.depths=depths
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            self.patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
        
        # Rest of the initialization code...
            if self.ape:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patches_resolution[0], self.patches_resolution[1]))
                trunc_normal_(self.absolute_pos_embed, std=.02)
        
            

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = TransformerBlock(
                dim=int(embed_dim * 2 ** i_layer),
               
                num_heads=num_heads[i_layer],
                
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
               
                )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
    
        x = self.patch_embed(x)

        if self.ape:
        # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(x.size(2), x.size(3)), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        dpr_index = 0

        for i in range(self.num_layers):
            layer = self.layers[i]
        # Use the current drop_path rate for this layer
            layer.drop_path = nn.Identity() if dpr[dpr_index] == 0. else DropPath(dpr[dpr_index])
            x = layer(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x = norm_layer(x)

                out = x.view(-1, self.patches_resolution[0], self.patches_resolution[1], self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

            dpr_index += self.depths[i]

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = up_conv(in_channels, out_channels)
    #self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        #coorAtt(out_channels),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    #x2 = self.att_block(x1, x2) 
    x1 = torch.cat((x2, x1), dim=1)
    x1 = self.conv_relu(x1)
    return x1
class Decoder1(nn.Module):
    def __init__(self, in_channels,  out_channels):
        super(Decoder1, self).__init__()
        self.up = up_conv(in_channels, out_channels)
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1):
        x1 = self.up(x1)
        x1 = self.conv_relu(x1)
        return x1

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class SwinUp(nn.Module):
    def __init__(self, dim):
        super(SwinUp, self).__init__()
        self.up = nn.Linear(dim, dim*2).cuda()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm(x)
        x = self.up(x)
        x = x.reshape(B, H, W, 2 * C)

        x0 = x[:,:,:,0:C//2]
        x1 = x[:,:,:,C//2:C]
        x2 = x[:,:,:,C:C+C//2]
        x3 = x[:,:,:,C+C//2:C*2]

        x0 = torch.cat((x0, x1), dim=1)
        x3 = torch.cat((x2, x3), dim=1)
        x = torch.cat((x0, x3), dim=2)

        #x = Variable(torch.randn(B, H * 2, W * 2, C // 2))

        x = x.reshape(B, -1, C // 2)
        return x



class SwinDecoder(nn.Module):

    def __init__(self,
                 embed_dim,
                 patch_size=4,
                 depths=2,
                 num_heads=6,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False):
        super(SwinDecoder, self).__init__()

        self.patch_norm = patch_norm

        # split image into non-overlapping patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        # build layers
        self.layer = TransformerBlock(
            dim=embed_dim//2,
            num_heads=num_heads,
            
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr,
            norm_layer=norm_layer,
            
           )
        
        self.up = up_conv(embed_dim, embed_dim//2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(embed_dim//2, embed_dim//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )



    def forward(self, x):
        """Forward function."""

        
        identity = x
        B, C, H, W = x.shape
        x = self.up(x) # B , C//2, 2H, 2W
        x = x.reshape(B , C//2, H*W*4)
        x = x.permute(0, 2, 1)

        #x_out, H, W, x = self.layer(x, H*2, W*2)

        x = x.permute(0, 2, 1)
        x = x.reshape(B , C//2, H, W)
        # B, C//4 2H, 2W
        x = self.conv_relu(x)

        return x
  

    

class Swin_Decoder(nn.Module):
  def __init__(self, in_channels, depths, num_heads):
    super(Swin_Decoder, self).__init__()
    self.up = SwinDecoder(in_channels, depths=depths, num_heads=num_heads)
    #self.up1 = nn.Upsample(scale_factor=2)
    #self.up2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, bias=True)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels//2, in_channels//4, kernel_size=1, stride=1, padding=0),
        nn.ReLU()
    )

  def forward(self, x1, x2):
    x1 = self.up(x1)
    #x1 = self.up2(x1)
    #x2 = self.att_block(x1, x2)
    x2 = self.conv2(x2)
    x1 = torch.cat((x2, x1), dim=1)
    out = self.conv_relu(x1)
    return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_heads = num_heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse




class UNet(nn.Module):
    def __init__(self, dim, n_class, in_ch=3):
        super().__init__()
        self.encoder = SwinTransformer(depths=[2, 2, 18, 2], num_heads=[ 4, 8, 16, 32 ], drop_path_rate=0.5, embed_dim=128)
        self.encoder2 = SwinTransformer(depths=[2, 2, 6, 2], num_heads=[ 3, 6, 12, 24 ], drop_path_rate=0.2, patch_size=8, embed_dim=96)
        #self.encoder.init_weights('checkpoints/swin_base_patch4_window7_224_22k.pth')
        #self.encoder2.init_weights('checkpoints/swin_tiny_patch4_window7_224.pth')
        self.layer1 = Swin_Decoder(8*dim, 2, 8)
        self.layer2 = Swin_Decoder(4*dim, 2, 4)
        self.layer3 = Swin_Decoder(2*dim, 2, 2)
        self.layer4 = Decoder1(dim, dim//2)
        self.layer5 = Decoder1(dim//2, dim//4)
        #self.layer4 = Decoder(dim, dim, dim//2)
        #self.layer5 = Decoder(dim//2, dim//2, dim//4)
        self.layer11 = Swin_Decoder(8 * dim, 2, 8)
        self.layer22 = Swin_Decoder(4 * dim, 2, 4)
        self.layer33 = Swin_Decoder(2 * dim, 2, 2)
        self.down1 = nn.Conv2d(in_ch, dim//4, kernel_size=1, stride=1, padding=0)
        self.down2 = conv_block(dim//4, dim//2)
        self.final = nn.Conv2d(dim//4, n_class, kernel_size=1, stride=1, padding=0)

        self.loss1 = nn.Sequential(
                nn.Conv2d(dim*8, n_class, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Upsample(scale_factor=32)
        )

        self.loss2 = nn.Sequential(
                nn.Conv2d(dim, n_class, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Upsample(scale_factor=4)
        )
        dim_s = 96
        dim_l = 128
        self.m1 = nn.Upsample(scale_factor=2)
        self.m2 = nn.Upsample(scale_factor=4)
        tb = dim_s + dim_l
        self.change1 = Conv_block(tb, dim)
        self.change2 = Conv_block(tb*2, dim*2)
        self.change3 = Conv_block(tb*4, dim*4)
        self.change4 = Conv_block(tb*8, dim*8)
        #self.cross_att_1 = Cross_Att(dim_s*1, dim_l*1)
        #self.cross_att_2 = Cross_Att(dim_s*2, dim_l*2)
        #self.cross_att_3 = Cross_Att(dim_s*4, dim_l*4)
        #self.cross_att_4 = Cross_Att(dim_s*8, dim_l*8)
        # Initialize AG modules with appropriate channel sizes
        self.scSE_final1 = scSE(dim_l * 1)  # 在初始化阶段定义scSE模块
        self.scSE_final2 = scSE(dim_l * 2)
        self.scSE_final3 = scSE(dim_l * 4)
        self.scSE_final4 = scSE(dim_l * 8)

  # Corresponding to e2 and d1


    def forward(self, x):
       out = self.encoder(x)
       out2 = self.encoder2(x)
       e1, e2, e3, e4 = out[0], out[1], out[2], out[3]
       #r1, r2, r3, r4 = out2[0], out2[1], out2[2], out2[3]
       #e1, r1 = self.cross_att_1(e1, r1)
       #e2, r2 = self.cross_att_2(e2, r2)
       #e3, r3 = self.cross_att_3(e3, r3)
       #e4, r4 = self.cross_att_4(e4, r4)
       #e1 = torch.cat([e1, self.m1(r1)], 1)
       #e2 = torch.cat([e2, self.m1(r2)], 1)
       #e3 = torch.cat([e3, self.m1(r3)], 1)
       #e4 = torch.cat([e4, self.m1(r4)], 1)
       #e1 = self.change1(e1)
       #e2 = self.change2(e2)
       #e3 = self.change3(e3)
       #e4 = self.change4(e4)
       #e1 = self.scSE_final1(e1)
       #e2 = self.scSE_final2(e2)
       #e3 = self.scSE_final3(e3)
       #e4 = self.scSE_final4(e4)
       loss1 = self.loss1(e4)
       ds1 = self.down1(x)
       ds2 = self.down2(ds1)
       #d1 = self.layer1(e4, self.layer11(e4, e3))  # Assuming d3 is output from previous decoder layer or input image
       #d2 = self.layer2(d1, self.layer22(d1, e2))  # Similarly adjust for d2
       #d3 = self.layer3(d2, self.layer33(d2, e1))  # Similarly adjust for d1
       d1 = self.layer1(e4, e3)
       d2 = self.layer2(d1, e2)
       d3 = self.layer3(d2, e1)
       loss2 = self.loss2(d3)
       d4 = self.layer4(d3)
       d5 = self.layer5(d4)
       #d4 = self.layer4(d3, ds2)
       #d5 = self.layer5(d4, ds1)
       o = self.final(d5)
       #return o
       return o, loss1, loss2


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(2, 3,64,64)).cuda()
    model = UNet(128, 1).cuda()
    print("Input shape:", x.shape)
    y = model(x)
    print('Output shape:',y[-1].shape)

