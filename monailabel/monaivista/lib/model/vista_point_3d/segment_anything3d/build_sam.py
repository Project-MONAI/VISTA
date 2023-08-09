# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import pdb
from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, SwinUNETR, SwinUNETR_P
def build_3d_samm_swin_b(checkpoint=None, in_channels=1, image_size=(96,96,96)):
    return _build_sam3d(
        encoder_embed_dim=12,
        checkpoint=checkpoint,
        in_channels=in_channels,
        image_size=image_size,
        depths=(2,2,2,2),
        num_heads=(3,6,12,24)
    )

def build_3d_samm_swin_l(checkpoint=None, in_channels=1, image_size=(96,96,96)):
    return _build_sam3d(
        encoder_embed_dim=48,
        checkpoint=checkpoint,
        in_channels=in_channels,
        image_size=image_size,
        depths=(2,2,2,2),
        num_heads=(3,6,12,24)
    )

def build_3d_samm_swin_l_nd(checkpoint=None, in_channels=1, image_size=(96,96,96)):
    return _build_sam3d(
        encoder_embed_dim=48,
        checkpoint=checkpoint,
        in_channels=in_channels,
        image_size=image_size,
        depths=(2,2,2,2),
        num_heads=(3,6,12,24),
        no_disc=True
    )

def build_3d_samm_swin_lh(checkpoint=None, in_channels=1, image_size=(96,96,96)):
    return _build_sam3d(
        encoder_embed_dim=48,
        checkpoint=checkpoint,
        in_channels=in_channels,
        image_size=image_size,
        depths=(2,2,2,2),
        num_heads=(3,6,12,24),
        use_hres=True
    )

def build_3d_samm_swin_h(checkpoint=None, in_channels=1, image_size=(96,96,96)):
    return _build_sam3d(
        encoder_embed_dim=48,
        checkpoint=checkpoint,
        in_channels=in_channels,
        image_size=image_size,
        depths=(2,2,18,2),
        num_heads=(3,6,12,24)
    )


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_3d_samm_swin_lp(
    encoder_embed_dim=48,
    checkpoint=None,
    in_channels=1,
    image_size=(96,96,96),
    prompt_embed_dim=256,
    depths=(2,2,2,2),
    num_heads=(3,6,12,24),        
):
    sam = SwinUNETR_P(
            img_size=image_size,
            in_channels=in_channels,
            out_channels=prompt_embed_dim,
            feature_size=encoder_embed_dim,
            depths=depths,
            num_heads=num_heads,
            use_v2=True,
        )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "swin_b": build_3d_samm_swin_b,
    "swin_l": build_3d_samm_swin_l,
    "swin_l_nd": build_3d_samm_swin_l_nd,
    "swin_lh": build_3d_samm_swin_lh,
    "swin_h": build_3d_samm_swin_h,
    "swin_lp": build_3d_samm_swin_lp
    
}

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam



def _build_sam3d(
    encoder_embed_dim,
    checkpoint=None,
    in_channels=1,
    image_size=(96,96,96),
    prompt_embed_dim=256,
    depths=(2,2,2,2),
    num_heads=(3,6,12,24),
    use_hres=False,
    no_disc = False
):
    if use_hres:
        image_embedding_size=(image_size[0]//2, image_size[1]//2, image_size[2]//2)
    else:
        image_embedding_size=(image_size[0]//4, image_size[1]//4, image_size[2]//4)
    sam = Sam(
        image_encoder = SwinUNETR(
            img_size=image_size,
            in_channels=in_channels,
            out_channels=prompt_embed_dim,
            feature_size=encoder_embed_dim,
            depths=depths,
            num_heads=num_heads,
            use_v2=True,
            use_hres=use_hres
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            input_image_size=image_size,
            image_embedding_size=image_embedding_size,
            use_hres=use_hres,
            no_disc=no_disc
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=0,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            use_hres=use_hres
        ),
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam
