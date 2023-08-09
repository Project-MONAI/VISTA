
from __future__ import annotations
from collections.abc import Sequence
import monai
import numpy as np
import torch
import torch.nn as nn
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from .sam_blocks import TwoWayTransformer, PositionEmbeddingRandom, MLP
import pdb
import time
rearrange, _ = optional_import("einops", name="rearrange")
NINF_VALUE= -99999
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

class UnetrUpBlock_noskip(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = self.conv_block(out)
        return out

class Point_Mapping_SAM(nn.Module):
    def __init__(
        self, feature_size,
        max_prompt=32
    ):
        super().__init__()
        transformer_dim = feature_size
        self.max_prompt = max_prompt
        self.feat_downsample = nn.Sequential(
            nn.Conv3d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.GELU(),
            nn.Conv3d(in_channels=feature_size, out_channels=transformer_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(feature_size))
        
        self.mask_downsample = nn.Conv3d(in_channels=2, out_channels=2,kernel_size=3, stride=2, padding=1)
        
        self.transformer = TwoWayTransformer(
                                depth=2,
                                embedding_dim=transformer_dim,
                                mlp_dim=512,
                                num_heads=4,
                            )
        self.pe_layer = PositionEmbeddingRandom(transformer_dim//2)
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, transformer_dim), nn.Embedding(1, transformer_dim)])
        self.not_a_point_embed = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(1, transformer_dim)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            nn.InstanceNorm3d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose3d(transformer_dim, transformer_dim, kernel_size=1, stride=1),
        )
        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim, 3)
        
    def forward(self, out, point_coords, point_labels):
        # downsample out
        out_low = self.feat_downsample(out)
        # embed points
        points = point_coords + 0.5  # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(points, out.shape[-3:])
        point_embedding[point_labels == -1] = 0.0
        point_embedding[point_labels == -1] += self.not_a_point_embed.weight
        point_embedding[point_labels == 0] += self.point_embeddings[0].weight
        point_embedding[point_labels == 1] += self.point_embeddings[1].weight
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(point_embedding.size(0), -1, -1)
        tokens_all = torch.cat((output_tokens, point_embedding), dim=1)
        # cross attention
        masks = []
        max_prompt = self.max_prompt
        for i in range(int(np.ceil(tokens_all.shape[0] / max_prompt))):
            idx = (i * max_prompt, min((i+1) * max_prompt, tokens_all.shape[0]))
            tokens = tokens_all[idx[0]:idx[1]]
            src = torch.repeat_interleave(out_low, tokens.shape[0], dim=0)
            pos_src = torch.repeat_interleave(self.pe_layer(out_low.shape[-3:]).unsqueeze(0), tokens.shape[0], dim=0)
            b, c, h, w, d = src.shape
            hs, src = self.transformer(src, pos_src, tokens)
            mask_tokens_out = hs[:, :1, :]
            src = src.transpose(1, 2).view(b, c, h, w, d)
            upscaled_embedding = self.output_upscaling(src)
            hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
            b, c, h, w, d = upscaled_embedding.shape
            masks.append((hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(b, -1, h, w, d))
        # plt.subplot(3,1,1);plt.imshow(masks[0,0,:,:,48].data.cpu().numpy());plt.subplot(3,1,2);plt.imshow(masks[1,0,:,:,48].data.cpu().numpy());plt.subplot(3,1,3);plt.imshow(masks[1,:,:,48].data.cpu().numpy());plt.savefig(f'{time.time()}.png')
        return torch.vstack(masks)
    

class Point_Mapping_GAP(nn.Module):
    def __init__(
        self, feature_size
    ):
        super().__init__()
        transformer_dim = feature_size
        self.feat_downsample = nn.Sequential(
            nn.Conv3d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.GELU(),
            nn.Conv3d(in_channels=feature_size, out_channels=transformer_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(feature_size))
        
        self.mask_downsample = nn.Conv3d(in_channels=2, out_channels=2,kernel_size=3, stride=2, padding=1)
        
        self.transformer = TwoWayTransformer(
                                depth=2,
                                embedding_dim=transformer_dim,
                                mlp_dim=512,
                                num_heads=4,
                            )
        self.pe_layer = PositionEmbeddingRandom(transformer_dim//2)
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, transformer_dim), nn.Embedding(1, transformer_dim)])
        self.not_a_point_embed = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(1, transformer_dim)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            nn.InstanceNorm3d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose3d(transformer_dim, transformer_dim, kernel_size=1, stride=1),
        )
        self.gap_mapping = nn.AdaptiveAvgPool3d((1,1,1))
        self.output_upscaling = UnetrUpBlock_noskip(
            spatial_dims=3,
            in_channels=transformer_dim,
            out_channels=transformer_dim,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim, 3)
        
    
    def forward(self, out, point_coords, point_labels, max_prompt=16):
        # downsample out
        out_low = self.feat_downsample(out)
        # embed points
        points = point_coords + 0.5  # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(points, out.shape[-3:])
        point_embedding[point_labels == -1] = 0.0
        point_embedding[point_labels == -1] += self.not_a_point_embed.weight
        point_embedding[point_labels == 0] += self.point_embeddings[0].weight
        point_embedding[point_labels == 1] += self.point_embeddings[1].weight
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(point_embedding.size(0), -1, -1)
        tokens = torch.cat((output_tokens, point_embedding), dim=1)
        # cross attention
        src = torch.repeat_interleave(out_low, tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(self.pe_layer(out_low.shape[-3:]).unsqueeze(0), tokens.shape[0], dim=0)
        b, c, h, w, d = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, :1, :]
        src = src.transpose(1, 2).view(b, c, h, w, d)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
        b, c, h, w, d = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(b, -1, h, w, d)
        # plt.subplot(3,1,1);plt.imshow(masks[0,0,:,:,48].data.cpu().numpy());plt.subplot(3,1,2);plt.imshow(masks[1,0,:,:,48].data.cpu().numpy());plt.subplot(3,1,3);plt.imshow(masks[1,:,:,48].data.cpu().numpy());plt.savefig(f'{time.time()}.png')
        return masks
    