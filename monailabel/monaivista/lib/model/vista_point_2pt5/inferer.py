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

from collections.abc import Callable, Sequence
from typing import Any
import numpy as np
import torch

from monai.apps.utils import get_logger
from monai.data.meta_tensor import MetaTensor
from monai.inferers import Inferer

import torch.nn.functional as F
from torch.cuda.amp import autocast
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
)
from monai.utils import (
    convert_to_dst_type,
)
from .utils.utils import prepare_sam_val_input

logger = get_logger(__name__)

class VISTASliceInferer(Inferer):
    def __init__(
        self,
        device: torch.device | str | None = None,
        progress: bool = False,
        cpu_thresh: int | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.progress = progress
        self.cpu_thresh = cpu_thresh

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        device: torch.device | str | None = None,
        n_z_slices: int = 9,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """

        device = kwargs.pop("device", self.device)

        if device is None and self.cpu_thresh is not None and inputs.shape[2:].numel() > self.cpu_thresh:
            device = "cpu"  # stitch in cpu memory if image is too large

        return vista_slice_inference(
            inputs,
            network,
            device,
            n_z_slices,
            *args,
            **kwargs,
        )
    
def vista_slice_inference(
    inputs: torch.Tensor | MetaTensor,
    predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
    device: torch.device | str | None = None,
    n_z_slices: int = 9,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)

    labels = kwargs.pop("labels")
    num_classes = len(labels)

    inputs_l = inputs
    pred_volume = torch.repeat_interleave(torch.zeros_like(inputs_l), num_classes+1, dim=1).float()

    inputs_l = inputs_l.squeeze()
    n_z_before_pad = inputs_l.shape[-1]
    # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice

    pd = (n_z_slices // 2, n_z_slices // 2)
    inputs_l = F.pad(inputs_l, pd, "constant", 0)

    computeEmbedding = kwargs.pop("computeEmbedding")

    if computeEmbedding:
        embedding = compute_embedding(n_z_slices, n_z_before_pad, inputs_l, predictor)
        return embedding

    post_pred = Compose([Activations(sigmoid=True)])
    post_pred_slice = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    class_prompts = kwargs.pop("class_prompts")
    point_prompts = kwargs.pop("point_prompts")
    cached_data = kwargs.pop("cached_data")
    cached_pred = cached_data["pred"] if cached_data else None

    cachedEmbedding = kwargs.pop("cachedEmbedding")
    cachedEmbedding = cachedEmbedding if cachedEmbedding else None
    original_affine = kwargs.pop("original_affine")

    if (class_prompts == None) and (point_prompts == None): 
        # Everything button: no class, no point prompts: iterate all slices
        class_prompts = [i for i in range(num_classes)]
        point_prompts = {'foreground': [], 'background': []}
        pred_volume = iterate_all(pred_volume, n_z_slices, n_z_before_pad, inputs_l, class_prompts, point_prompts, predictor, post_pred, cachedEmbedding, cached_pred, device)
    elif (point_prompts == None) and (class_prompts != None):
        if class_prompts:
            # class prompts only: need to iterate all slices
            point_prompts = {'foreground': [], 'background': []}
            pred_volume = iterate_all(pred_volume, n_z_slices, n_z_before_pad, inputs_l, class_prompts, point_prompts, predictor, post_pred, cachedEmbedding, cached_pred, device)
        else:
            pred_volume = pred_volume.argmax(1).unsqueeze(1)
    elif (class_prompts == None) and (point_prompts != None):
        class_prompts = []
        pred_volume = update_slice(pred_volume, n_z_slices, n_z_before_pad, inputs_l, class_prompts, point_prompts, predictor, post_pred_slice, cached_pred, num_classes, original_affine, device)
    else:
        pred_volume = update_slice(pred_volume, n_z_slices, n_z_before_pad, inputs_l, class_prompts, point_prompts, predictor, post_pred_slice, cached_pred, num_classes, original_affine, device)

    if temp_meta is not None:
        final_output = convert_to_dst_type(pred_volume, temp_meta, device=device)[0]
    else:
        final_output = convert_to_dst_type(pred_volume, inputs, device=device)[0]

    return final_output  # type: ignore


def compute_embedding(n_z_slices, n_z_before_pad, inputs_l, predictor):
    # image_embedding_dict saves the image embedding for each slice.
    # The key (int) is the index of center slice in original volume (before padding), e.g., 0,1,2,...n if the
    # original volume has n slices.
    # The value (torch.tensor) is the corresponding image embedding.
    image_embedding_dict = {}
    # get image embedding from the predictor (network) forward function
    for start_idx in range((n_z_slices // 2), (n_z_slices // 2 + n_z_before_pad)):
        inputs = inputs_l[..., start_idx - (n_z_slices // 2):start_idx + (n_z_slices // 2) + 1].permute(2, 0, 1)
        # Here, the batch size is 1 (it is possible to increase batch size if the device has enough memory).
        data = [{"image": inputs}]
        with autocast():
            image_embeddings = predictor.get_image_embeddings(data)  # (1, C, H, W)
        # Save image embedding for each slice to RAM
        image_embedding_dict[start_idx - (n_z_slices // 2)] = image_embeddings.cpu()

    return image_embedding_dict

def update_slice(pred_volume, n_z_slices,n_z_before_pad, inputs_l, class_prompts, point_prompts, predictor, post_pred_slice, cached_pred, num_classes, original_affine, device):
    z_indices = [p[2]+(9//2) for p in point_prompts["foreground"]]
    z_indices.extend([p[2]+(9//2) for p in point_prompts["background"]])
    z_indices = list(set(z_indices))

    pred_volume = pred_volume.argmax(1).unsqueeze(1)

    for start_idx in z_indices:
        if start_idx < (n_z_slices // 2):
            continue

        inputs = inputs_l[..., start_idx - (n_z_slices // 2): start_idx + (n_z_slices // 2) + 1].permute(2, 0, 1)
        data, unique_labels = prepare_sam_val_input(inputs.cuda(), class_prompts, point_prompts, start_idx, original_affine)

        predictor.eval()
        with torch.cuda.amp.autocast():
            outputs = predictor(data)
            logit = outputs[0]["high_res_logits"]

        out_list = torch.unbind(logit, dim=0)
        y_pred = torch.stack(post_pred_slice(out_list)).float()

        pred_volume = pred_volume.float()
        idx = torch.where(y_pred[0] == 1)
        z_idx = start_idx - (n_z_slices // 2)

        if cached_pred is not None:
            if class_prompts:
                cached_pred_idx = torch.where(cached_pred[:, :, :, z_idx] == class_prompts[0] + 1)
                cached_pred[:, :, :, z_idx][cached_pred_idx] = 0
                cached_pred[:, :, :, z_idx][idx] = class_prompts[0] + 1
            else:
                cached_pred[:, :, :, z_idx][idx] = num_classes + 1
        else:
            pred_volume[0, :, :, :, z_idx][idx] = class_prompts[0] + 1 if class_prompts else num_classes + 1

    if cached_pred is not None:
        pred_volume[0] = cached_pred.float()

    return pred_volume

def iterate_all(pred_volume, n_z_slices, n_z_before_pad, inputs_l, class_prompts, point_prompts, predictor, post_pred, cachedEmbedding, cached_pred, device):
    start_range = range(n_z_slices // 2, min((n_z_slices // 2 + n_z_before_pad), len(cachedEmbedding))) if cachedEmbedding else range(n_z_slices // 2, n_z_slices // 2 + n_z_before_pad)
    for start_idx in start_range:
        inputs = inputs_l[..., start_idx - n_z_slices // 2:start_idx + n_z_slices // 2 + 1].permute(2, 0, 1)
        data, unique_labels = prepare_sam_val_input(inputs.cuda(), class_prompts, point_prompts, start_idx)
        predictor = predictor.eval()
        with autocast():
            if cachedEmbedding:
                curr_embedding = cachedEmbedding[start_idx].cuda()
                outputs = predictor.get_mask_prediction(data, curr_embedding)
            else:
                outputs = predictor(data)
            logit = outputs[0]["high_res_logits"]
        
        out_list = torch.unbind(logit, dim=0)
        y_pred = torch.stack(post_pred(out_list)).float()
        pred_idx = start_idx - (n_z_slices // 2) if not cachedEmbedding else start_idx
        pred_volume[0, unique_labels, ..., pred_idx] = y_pred

    pred_volume = pred_volume.argmax(1).unsqueeze(1).cpu()
    pred_volume = pred_volume.float()

    if cached_pred is not None:
        pred_volume_idx = torch.where(pred_volume[0] != 0)
        cached_pred[pred_volume_idx] = pred_volume[0][pred_volume_idx]
        pred_volume[0] = cached_pred.float()

    return pred_volume


