# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import pdb
from functools import partial
import monai
from monai.networks.nets import SegResNetDS, SwinUNETR, SegResNet
from .modeling import VISTA3D, Point_Mapping_SAM, Class_Mapping_Vanila, Point_Mapping_GAP

def build_vista3d_segresnet_vanila(
    encoder_embed_dim=48,
    in_channels=1,
    image_size=(96,96,96)  
):
    segresnet = SegResNetDS(
            in_channels=in_channels,
            blocks_down=(1, 2, 2, 4, 4),
            norm="instance",
            out_channels=encoder_embed_dim,
            init_filters=encoder_embed_dim,
            dsdepth=1
        )
    point_head = Point_Mapping_SAM(
        feature_size=encoder_embed_dim
    )
    class_head = Class_Mapping_Vanila(
        n_classes=512, feature_size=encoder_embed_dim, use_mlp=True
    )
    vista = VISTA3D(
        image_encoder=segresnet,
        class_head=class_head,
        point_head=point_head,
        feature_size=encoder_embed_dim
    )
    return vista


def build_vista3d_swinunetr_vanila(
    encoder_embed_dim=48,
    in_channels=1,
    image_size=(96,96,96),
    depths=(2,2,2,2),
    num_heads=(3,6,12,24),        
):
    swinunetr = SwinUNETR(
            img_size=image_size,
            in_channels=in_channels,
            out_channels=encoder_embed_dim,
            feature_size=encoder_embed_dim,
            depths=depths,
            num_heads=num_heads,
            use_v2=True,
        )
    point_head = Point_Mapping_SAM(
        feature_size=encoder_embed_dim
    )
    class_head = Class_Mapping_Vanila(
        n_classes=512, feature_size=encoder_embed_dim
    )
    vista = VISTA3D(
        image_encoder=swinunetr,
        class_head=class_head,
        point_head=point_head,
        feature_size=encoder_embed_dim
    )
    return vista

def build_vista3d_swinunetr_gap(
    encoder_embed_dim=48,
    in_channels=1,
    image_size=(96,96,96),
    depths=(2,2,2,2),
    num_heads=(3,6,12,24),        
):
    swinunetr = SwinUNETR(
            img_size=image_size,
            in_channels=in_channels,
            out_channels=encoder_embed_dim,
            feature_size=encoder_embed_dim,
            depths=depths,
            num_heads=num_heads,
            use_v2=True,
        )
    point_head = Point_Mapping_GAP(
        feature_size=encoder_embed_dim
    )
    class_head = Class_Mapping_Vanila(
        n_classes=512, feature_size=encoder_embed_dim
    )
    vista = VISTA3D(
        image_encoder=swinunetr,
        class_head=class_head,
        point_head=point_head,
        feature_size=encoder_embed_dim
    )
    return vista



vista_model_registry = {
    "vista3d_segresnet": build_vista3d_segresnet_vanila,
    "vista3d_swinunetr": build_vista3d_swinunetr_vanila,
    "vista3d_swinunetr_gap": build_vista3d_swinunetr_gap
}

