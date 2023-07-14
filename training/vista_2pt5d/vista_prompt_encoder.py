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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from segment_anything.modeling.common import LayerNorm2d
from segment_anything.modeling.prompt_encoder import PromptEncoder


class VistaPromptEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        n_classes: int = 512,
        clip_class_label_prompt: bool = False,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
          n_classes (int): The number of pre-defined classes.
          clip_class_label_prompt (bool): Using clip txt features
            as class label prompt.
        """
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)

        self.clip_class_label_prompt = clip_class_label_prompt
        # Add support for onehot vector embedding for pre-defined classes
        if self.clip_class_label_prompt:
            raise NotImplementedError
        else:
            self.label_embeddings = nn.Embedding(n_classes, embed_dim)
        self.no_label_embed = nn.Embedding(1, embed_dim)

    def _embed_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Embeds onehot vector inputs."""
        if self.clip_class_label_prompt:
            raise NotImplementedError
        else:
            # Add support for onehot vector embedding for pre-defined classes
            label_embedding = self.label_embeddings(labels)
        return label_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif labels is not None:
            return labels.shape[0]
        else:
            return 1

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        class_labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed
          class_labels (torch.Tensor or none): labels to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks, class_labels)

        # Add support for onehot vector embedding for pre-defined classes
        if class_labels is not None:
            label_embeddings = self._embed_labels(class_labels)
        else:
            label_embeddings = self.no_label_embed.weight.reshape(1, 1, -1).expand(bs, -1, -1)

        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())

        # Add support for onehot vector embedding for pre-defined classes
        sparse_embeddings = torch.cat([sparse_embeddings, label_embeddings], dim=1)

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
