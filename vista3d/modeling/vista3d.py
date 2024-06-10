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
import copy
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
from monai.metrics import compute_dice
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from scripts.monai_trans_utils import get_largest_connected_component_mask as lcc
from scripts.utils import convert_points_to_disc, get_next_points_auto_point, get_next_points, sample_points_patch_val, debug_ccp
import pdb
import time

rearrange, _ = optional_import("einops", name="rearrange")
NINF_VALUE= -9999
PINF_VALUE = 9999

class VISTA3D2(nn.Module):
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
        self.point_freeze = False
    
    def precompute_embedding(self, input_images):
        """ precompute image embedding, require sliding window inference
        """
        raise NotImplementedError
    
    def clear_cache(self):
        self.image_embeddings = None

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
            # also remove padded points, mainly used for inference.
            not_pad_indices = (point_labels != -1).any(0)
            point_coords = point_coords[:, not_pad_indices]
            point_labels = point_labels[:, not_pad_indices]
        else:
            point_coords = None
            point_labels = None        
        return point_coords, point_labels

    def connected_components_combine(self, logits, point_logits, point_coords, point_labels, mapping_index, thred=0.5):
        """ Combine auto results with point click response, or combine previous mask with point click response. 
        For mapping_index with point clicks, NaN values in logits will be replaced with point_logits. Meanwhile, the added/removed
        region in point clicks must be updated by the lcc function. Notice, if a positive point is within logits/prev_mask, the components containing the positive point
        will be added.
        """
        logits = logits.as_tensor() if isinstance(logits, monai.data.MetaTensor) else logits 
        _logits = logits[mapping_index]
        inside = []
        for i in range(_logits.shape[0]):
            inside.append(np.any([_logits[i, 0, round(p[0].item()), round(p[1].item()), round(p[2].item())].item() > 0 for p in point_coords[i]]))
        inside = torch.tensor(inside).to(logits.device)
        nan_mask = torch.isnan(_logits)
        _logits = torch.nan_to_num(_logits, nan=NINF_VALUE).sigmoid()
        pos_region = point_logits.sigmoid() > thred
        diff_pos = torch.logical_and(torch.logical_or((_logits <= thred), inside.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), pos_region)
        diff_neg = torch.logical_and((_logits > thred), ~pos_region)
        cc = lcc(diff_pos, diff_neg, point_coords=point_coords, point_labels=point_labels)
        # cc is the region that can be updated by point_logits.
        cc = cc.to(logits.device)
        # Need to replace NaN with point_logits. diff_neg will never lie in nan_mask, only remove unconnected positive region.
        uc_pos_region = torch.logical_and(pos_region, ~cc)
        fill_mask = torch.logical_and(nan_mask, uc_pos_region)
        if fill_mask.any():
            # fill in the mean negative value
            point_logits[fill_mask] = -1
        # replace logits nan value and cc with point_logits
        cc = torch.logical_or(nan_mask, cc).to(logits.dtype)
        logits[mapping_index] *= (1 - cc)
        logits[mapping_index] += cc * point_logits
        # debug_ccp(_logits, point_logits.sigmoid(), point_coords, point_labels, diff, cc, logits[mapping_index], np.random.randint(10000))
        return logits
    
    def gaussian_combine(self, logits, point_logits, point_coords, point_labels, mapping_index, radius):
        if radius is None:
            radius = min(point_logits.shape[-3:]) // 5 # empirical value 5
        weight = 1 - convert_points_to_disc(point_logits.shape[-3:], point_coords, point_labels, radius=radius).sum(1, keepdims=True)
        weight[weight < 0] = 0
        logits = logits.as_tensor() if isinstance(logits, monai.data.MetaTensor) else logits
        logits[mapping_index] *= weight
        logits[mapping_index] += (1 - weight) * point_logits
        return logits
    
    def set_auto_grad(self, auto_freeze = False, point_freeze=False):
        if auto_freeze != self.auto_freeze:
            if hasattr(self.image_encoder, 'set_auto_grad'):
                self.image_encoder.set_auto_grad(auto_freeze = auto_freeze, point_freeze=point_freeze)
            else:
                for param in self.image_encoder.parameters():
                    param.requires_grad = (not auto_freeze) and (not point_freeze)
            for param in self.class_head.parameters():
                param.requires_grad = not auto_freeze
            self.auto_freeze = auto_freeze

        if point_freeze != self.point_freeze:
            if hasattr(self.image_encoder, 'set_auto_grad'):
                self.image_encoder.set_auto_grad(auto_freeze = auto_freeze, point_freeze=point_freeze)
            else:
                for param in self.image_encoder.parameters():
                    param.requires_grad = (not auto_freeze) and (not point_freeze)
            for param in self.point_head.parameters():
                param.requires_grad = not point_freeze
            self.point_freeze = point_freeze

    def forward(self, input_images,       
        point_coords=None,
        point_labels=None,
        class_vector=None,
        prompt_class=None,
        patch_coords=None,
        labels=None,
        label_set=None,
        prev_mask=None,
        radius=None,
        val_point_sampler=sample_points_patch_val,
        **kwargs):
        image_size = input_images.shape[-3:]
        device = input_images.device

        bs = self.get_bs(class_vector, point_coords)
        if patch_coords is not None and point_coords is not None:
            """ patch_coords is passed from monai_utils.sliding_window_inferer. 
            """
            ## Automatic point sample in validation
            if labels is not None and label_set is not None:
                # if labels is not None, sample from labels for each patch.
                point_coords, point_labels, prompt_class = val_point_sampler(labels, patch_coords, label_set, prev_mask, class_vector)
                if prompt_class[0].item() == 0:
                    point_labels[0] = -1
                labels, prev_mask = None, None
            ## User provided click points in inference
            else:
                point_coords, point_labels = self.update_point_to_patch(patch_coords, point_coords, point_labels)
        
        if point_coords is not None and point_labels is not None:
            # remove points that used for padding purposes (point_label = -1)
            mapping_index = ((point_labels != -1).sum(1) > 0).to(torch.bool)
            if mapping_index.any():
                point_coords = point_coords[mapping_index]
                point_labels = point_labels[mapping_index]
                if prompt_class is not None:
                    prompt_class = prompt_class[mapping_index]
            else:
                if self.auto_freeze or (class_vector is None and patch_coords is None):
                    # if auto_freeze, point prompt must exist to allow loss backward
                    # in training, class_vector and point cannot both be None due to loss.backward()
                    mapping_index.fill_(True)
                else:
                    point_coords, point_labels = None, None

        if point_coords is None and class_vector is None:
            return (NINF_VALUE + torch.zeros([bs,1,*image_size], device=device))
        

        if self.image_embeddings is not None and kwargs.get('keep_cache', False) and class_vector is None:
            out, out_auto = self.image_embeddings, None
        else:
            out, out_auto = self.image_encoder(input_images, with_point=point_coords is not None or kwargs.get('merge_with_trial', False), with_label=class_vector is not None)
        input_images = None

        
        # force releasing memories that set to None
        torch.cuda.empty_cache()

        if class_vector is not None:
            logits, _ = self.class_head(out_auto, class_vector)
            if kwargs.get('merge_with_trial', False):
                logits = self.point_head_iterative_trial(logits, labels[patch_coords], out, point_coords, point_labels, class_vector[0], prompt_class[0], n_trials=3)              
            if point_coords is not None:
                point_logits = self.point_head(out, point_coords, point_labels, class_vector=prompt_class)
                if patch_coords is None:
                    # during training, using gaussian ball combine
                    logits = self.gaussian_combine(logits, point_logits, point_coords, point_labels, mapping_index, radius)
                else:
                    # during validation use largest component
                    logits = self.connected_components_combine(logits, point_logits, point_coords, point_labels, mapping_index)
        else:    
            logits = NINF_VALUE + torch.zeros([bs,1,*image_size], device=device, dtype=out.dtype)
            logits[mapping_index] = self.point_head(out, point_coords, point_labels, class_vector=prompt_class)
            if prev_mask is not None and patch_coords is not None:
                logits = self.connected_components_combine(prev_mask[patch_coords].transpose(1,0).to(logits.device), logits[mapping_index], point_coords, point_labels, mapping_index)

        if kwargs.get('keep_cache', False) and class_vector is None:
            self.image_embeddings = out.detach()            
        return logits

    @torch.no_grad()
    def point_head_iterative_trial(self, logits, labels, out, point_coords, point_labels, class_vector, prompt_class, n_trials=3):
        """ The prompt class is the local label set while class vector is the mapped global label set
        """
        logits_update = logits.detach().clone()
        for trial_idx in range(n_trials):
            if trial_idx == 0: 
                point_coords, point_labels = get_next_points_auto_point(logits> 0, labels, prompt_class, class_vector, use_fg=True)
            else:
                point_coords, point_labels = get_next_points_auto_point(logits> 0, labels, prompt_class, class_vector, use_fg=False)
            mapping_index = ((point_labels != -1).sum(1) > 0).to(torch.bool)
            point_coords = point_coords[mapping_index]
            point_labels = point_labels[mapping_index]
            if (torch.sum(mapping_index) == 1 and mapping_index[0]) or torch.sum(mapping_index) == 0 :
                return logits
            if trial_idx == 0:
                best_dice = []
                for i in range(len(prompt_class)):
                    dice = compute_dice(y_pred= (logits[[i]] > 0).to(labels.device), y=labels==prompt_class[i]).item()
                    if np.isnan(dice):
                        best_dice.append(- (logits[[i]] > 0).sum())
                    else:
                        best_dice.append(dice)

            point_logits = self.point_head(out, point_coords, point_labels, class_vector=class_vector[mapping_index])

            target_logits = self.connected_components_combine(logits, point_logits, point_coords, point_labels, mapping_index)
            combine_dice = []
            for i in range(len(prompt_class)):
                if mapping_index[i]:
                    dice = compute_dice(y_pred=(target_logits[[i]] > 0).to(labels.device), y = (labels==prompt_class[i])).item()
                    if np.isnan(dice):
                        combine_dice.append(- (target_logits[[i]] > 0).sum())
                    else:
                        combine_dice.append(dice)
                else:
                    combine_dice.append(-1)
            # check the dice for each label
            for i in range(len(prompt_class)):
                if prompt_class[i] == 0:
                    continue
                if combine_dice[i] > best_dice[i]:
                    # print(trial_idx, prompt_class[i], combine_dice[i], best_dice[i])
                    logits_update[i] = copy.deepcopy(target_logits[i])
                    best_dice[i] = copy.deepcopy(combine_dice[i])

        labels, target_logits, logits, best_dice, combine_dice = None, None, None, None, None
        # force releasing memories that set to None
        torch.cuda.empty_cache()
        return logits_update
