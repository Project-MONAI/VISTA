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
from lib.model.vista_point_2pt5.inferer import VISTASliceInferer
from monai.inferers import Inferer
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
)
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.transform.post import Restored


class VISTAPOINT2PT5(BasicInferTask):
    """
    This provides Inference Engine for pre-trained spleen segmentation (UNet) model over MSD Dataset.
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

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            ScaleIntensityRanged(keys="image", a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True),
        ]

    def inferer(self, data=None) -> Inferer:
        return VISTASliceInferer()

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Restored(keys="pred", ref_image="image"),
        ]
