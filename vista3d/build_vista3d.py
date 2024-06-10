# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import pdb
from functools import partial
import monai
from .modeling import VISTA3D2, Point_Mapping_SAM, Class_Mapping_Classify, SegResNetDS2

def build_vista3d_segresnet_decoder(
    encoder_embed_dim=48,
    in_channels=1,
    image_size=(96,96,96)  
):
    segresnet = SegResNetDS2(
            in_channels=in_channels,
            blocks_down=(1, 2, 2, 4, 4),
            norm="instance",
            out_channels=encoder_embed_dim,
            init_filters=encoder_embed_dim,
            dsdepth=1
        )
    point_head = Point_Mapping_SAM(
        feature_size=encoder_embed_dim,
        last_supported=132
    )
    class_head = Class_Mapping_Classify(
        n_classes=512, feature_size=encoder_embed_dim, use_mlp=True
    )
    vista = VISTA3D2(
        image_encoder=segresnet,
        class_head=class_head,
        point_head=point_head,
        feature_size=encoder_embed_dim
    )
    return vista

vista_model_registry = {
    "vista3d_segresnet_d": build_vista3d_segresnet_decoder
}