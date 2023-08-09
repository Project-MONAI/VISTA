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
import monai
import torch
import torch.nn.functional as F
from monai.apps.utils import get_logger
from monai.data.meta_tensor import MetaTensor
from monai.inferers import Inferer
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import convert_to_dst_type
from torch.cuda.amp import autocast
from monai.utils import optional_import
from functools import partial

rearrange, _ = optional_import("einops", name="rearrange")
import numpy as np
from .utils import point_based_window_inferer, pad_previous_mask
from .monai_utils import sliding_window_inference


logger = get_logger(__name__)

def infer_wrapper(inputs, model, **kwargs):
    outputs = model(input_images=inputs,**kwargs)
    return outputs.transpose(1,0)

class VISTAPOINT3DInferer(Inferer):
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
        point_prompts = kwargs.pop("point_prompts")
        class_prompts = kwargs.pop("class_prompts")
        print("point_prompts: {}".format(point_prompts))

        if device is None and self.cpu_thresh is not None and inputs.shape[2:].numel() > self.cpu_thresh:
            device = "cpu"  # stitch in cpu memory if image is too large


        network = network.eval()



        if point_prompts is not None:
            point = self.transform_points(point_prompts, np.linalg.inv(inputs['image'].affine[0]) @ inputs['image'].meta['original_affine'][0].numpy())
            vista_sliding_window_inferer = point_based_window_inferer
        else:
            vista_sliding_window_inferer = sliding_window_inference


        point_coords = None
        point_label = None

        label_prompt  = [i+1 for i in class_prompts]
        # label_prompt = [50]

        point = None
        self.prev_mask = None

        pred = vista_sliding_window_inferer(	
            inputs=inputs,	
            roi_size=[96, 96, 96],	
            sw_batch_size=1,	
            predictor=partial(infer_wrapper, model=network),	
            mode="gaussian",	
            overlap=0.25,	
            sw_device=device,	
            device=device,
            point_coords=torch.tensor(point).to(device) if point is not None else None,
            point_labels=torch.tensor(point_label).to(device) if point_label is not None else None,
            class_vector=torch.tensor(label_prompt).to(device) if label_prompt is not None else None,
            masks=torch.tensor(self.prev_mask).to(device) if self.prev_mask is not None else None,
            point_mask=None
        )

        # if not hasattr(inputs["pred"],'meta'):
        #     inputs["pred"] = monai.data.MetaTensor(inputs["pred"], affine=inputs["image"].meta["affine"], meta=inputs["image"].meta)

        # post_pred = Compose([Activations(sigmoid=True)])
        post_pred_thresh = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        out_list = torch.unbind(pred, dim=0)
        y_pred = torch.stack(post_pred_thresh(out_list)).float()

        return y_pred


    def transform_points(self, point, affine, original_affine=None):
        """ transform point to the coordinates of the transformed image
        point: numpy array [bs, N, 3]
        """
        bs, N = point.shape[:2]

        point = np.concatenate((point,np.ones((bs, N,1))), axis=-1)
        point = rearrange(point, 'b n d -> d (b n)')
        point = affine @ point
        point = rearrange(point, 'd (b n)-> b n d', b=bs)[:,:,:3]
        return point
        
