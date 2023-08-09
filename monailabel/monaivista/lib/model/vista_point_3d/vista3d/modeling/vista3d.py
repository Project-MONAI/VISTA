# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import itertools
from collections.abc import Sequence
import monai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from matplotlib import pyplot as plt
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from scripts.utils import convert_points_to_disc, generate_prompt_pairs_val
import pdb
import time

rearrange, _ = optional_import("einops", name="rearrange")
NINF_VALUE= -99999

class VISTA3D(nn.Module):
    def __init__(
            self,
            image_encoder,
            class_head,
            point_head,
            feature_size
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.class_head = class_head
        self.point_head = point_head
        self.image_embeddings = None  
        self.weight_mapper = nn.Sequential(
            nn.Linear(feature_size, 4*feature_size),
            nn.GELU(),
            nn.InstanceNorm1d(4*feature_size),
            nn.Linear(4*feature_size, 1)
        )
        self.auto_freeze = False
    
    def precompute_embedding(self, input_images):
        """ precompute image embedding, require sliding window inference
        """
        raise NotImplementedError
    
    def clear_cache(self):
        pass

    def get_bs(self, class_vector, point_coords):
        if class_vector is None:
            assert point_coords is not None, 'prompt is required'
            return point_coords.shape[0]
        else:
            return class_vector.shape[0]
        
    def update_point_to_patch(self, patch_coords, point_coords, point_labels):
        """ Update point_coords with respect to patch coords. 
            If point is outside of the patch, remove the coordinates and set label to -1
        """
        patch_ends = [patch_coords[-3].stop, patch_coords[-2].stop, patch_coords[-1].stop]
        patch_starts = [patch_coords[-3].start, patch_coords[-2].start, patch_coords[-1].start]
        # update point coords
        patch_starts = torch.tensor(patch_starts, device=point_coords.device).unsqueeze(0).unsqueeze(0)
        patch_ends = torch.tensor(patch_ends, device=point_coords.device).unsqueeze(0).unsqueeze(0)
        # [1 N 1]
        indices = torch.logical_and(((point_coords - patch_starts) > 0).all(2), ((patch_ends - point_coords) > 0).all(2))
        # check if it's within patch coords
        point_coords = point_coords.clone() - patch_starts
        point_labels = point_labels.clone()
        if indices.any():
            point_labels[~indices] = -1
            point_coords[~indices] = 0
        else:
            point_coords = None
            point_labels = None        
        return point_coords, point_labels
    
    def set_auto_grad(self, requires_grad = True):
        self.auto_freeze = not requires_grad
        for param in self.image_encoder.parameters():
            param.requires_grad = requires_grad
        for param in self.class_head.parameters():
            param.requires_grad = requires_grad

    def forward(self, input_images,       
        point_coords=None,
        point_labels=None,
        class_vector=None,
        patch_coords=None,
        labels=None,
        label_set=None,
        radius=None,
        **kwargs):

        # Calculate image features
        if self.image_embeddings is not None:
            raise NotImplementedError
        else:
            out = self.image_encoder(input_images)

        bs = self.get_bs(class_vector, point_coords)
        label_logits = 0
        point_logits = 0       
        
        if patch_coords is not None and point_coords is not None:
            """ patch_coords is passed from monai_utils.sliding_window_inferer. 
            """
            if labels is not None and label_set is not None:
                # if labels is not None, sample from labels for each patch.
                # only in validation when labels of the whole image is provided, sample points for every position
                _, point_coords, point_labels, _, _ = \
                    generate_prompt_pairs_val(labels[patch_coords], label_set, 
                                            image_size=input_images.shape[-3:],
                                            max_point=1,
                                            device=labels.device)
            else:
                point_coords, point_labels = self.update_point_to_patch(patch_coords, point_coords, point_labels)
        
        if point_coords is not None and point_labels is not None:
            # remove points that used for padding purposes (point_label = -1)
            mapping_index = ((point_labels != -1).sum(1) > 0).to(torch.bool)
            if mapping_index.any():
                point_coords = point_coords[mapping_index]
                point_labels = point_labels[mapping_index]
            else:
                if self.auto_freeze:
                    # if auto_freeze, point prompt must exist to allow loss backward
                    mapping_index.fill_(True)
                else:
                    point_coords, point_labels = None, None

        if point_coords is None and class_vector is None:
            return (NINF_VALUE + torch.zeros([bs,1,input_images.shape[-3],input_images.shape[-2],input_images.shape[-1]], device=input_images.device))
        
        if class_vector is not None:
            label_logits, class_embedding = self.class_head(out, class_vector)
            if point_coords is not None:
                weight = torch.ones(label_logits.shape, device=point_coords.device)
                point_logits = self.point_head(out, point_coords, point_labels)
                if radius is None:
                    radius = min(point_logits.shape[-3:]) // 5 # empirical value 5
                weight[mapping_index] = 1 - convert_points_to_disc(point_logits.shape[-3:], point_coords, point_labels, radius=radius).sum(1, keepdims=True)
                weight[weight < 0] = 0
                logits = ((1-weight) * (1-mapping_index.to(float)).view(-1,1,1,1,1) + weight) * label_logits 
                logits = logits.as_tensor() if isinstance(logits, monai.data.MetaTensor) else logits
                logits[mapping_index] += (1-weight[mapping_index]) * point_logits
            else:
                logits = label_logits
        else:    
            logits = NINF_VALUE + torch.zeros([bs,1,input_images.shape[-3],input_images.shape[-2],input_images.shape[-1]], device=input_images.device, dtype=out.dtype)
            logits[mapping_index] = self.point_head(out, point_coords, point_labels)
        return logits