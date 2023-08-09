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

from typing import Callable, Sequence

from lib.basic_infer import BasicInferTask
from lib.model.vista_point_3d.inferer import VISTAPOINT3DInferer
from monai.inferers import Inferer
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    CastToTyped,
    Invertd,
    Activationsd,
    AsDiscreted,
)
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.transform.post import Restored
import torch
from lib.transforms.transforms import ThreshMergeLabeld

class VISTAPOINT3D(BasicInferTask):
    """
    This provides Inference Engine for pre-trained VISTA segmentation model.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.5, 1.5, 1.5),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        description="A pre-trained model for volumetric (2.5D) segmentation of the monai vista",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.target_spacing = target_spacing

    def is_valid(self) -> bool:
        return True

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityRanged(keys="image", a_min=-963.8247715525971, a_max=1053.678477684517, b_min=0.0, b_max=1.0, clip=True),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=(1.5,1.5,1.5), mode="bilinear", align_corners=True),
            CastToTyped(keys="image", dtype=torch.float32),
        ]

    def inferer(self, data=None) -> Inferer:
        return VISTAPOINT3DInferer(device=data.get("device") if data else None)

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            # Invertd(	
            #     keys="pred",	
            #     transform=self.infer_transforms,	
            #     orig_keys="image",	
            #     meta_keys="pred_meta_dict",	
            #     orig_meta_keys="image_meta_dict",	
            #     meta_key_postfix="meta_dict",	
            #     nearest_interp=False,	
            #     to_tensor=True
            # ),	
            # Activationsd(	
            #     keys="pred",	
            #     softmax=False,	
            #     sigmoid=True
            # ),
            # AsDiscreted(
            #     keys="pred",
            #     threshold=0.5
            # ),
            ThreshMergeLabeld(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
