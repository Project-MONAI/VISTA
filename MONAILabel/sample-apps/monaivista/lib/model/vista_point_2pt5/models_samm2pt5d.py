# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple
from functools import partial

from .segment_anything.modeling.image_encoder import ImageEncoderViT
from .segment_anything.modeling.mask_decoder import MaskDecoder
from .segment_anything.modeling.prompt_encoder import PromptEncoder
from .segment_anything.modeling import TwoWayTransformer
import monai


class Samm2pt5D(nn.Module):
    mask_threshold: float = 0.5
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def get_image_embeddings(self, batched_input: List[Dict[str, Any]], ):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)
        return image_embeddings

    def get_mask_prediction(self, batched_input: List[Dict[str, Any]], image_embeddings,
                            multimask_output: bool = False):

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
                # raise NotImplementedError
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
                class_labels=image_record.get("labels", None)
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            high_res_masks = self.postprocess_masks(
                low_res_masks,
                # input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = high_res_masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "high_res_logits": high_res_masks,

                }
            )
        return outputs

    def forward(
            self,
            batched_input: List[Dict[str, Any]],
            multimask_output: bool = False,
            is_train: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
                # raise NotImplementedError
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
                class_labels=image_record.get("labels", None)
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            if is_train:
                outputs.append(
                    {
                        "iou_predictions": iou_predictions,
                        "low_res_logits": low_res_masks,
                    }
                )
            else:
                high_res_masks = self.postprocess_masks(
                    low_res_masks,
                    # input_size=image_record["image"].shape[-2:],
                    original_size=image_record["original_size"],
                )
                masks = high_res_masks > self.mask_threshold
                outputs.append(
                    {
                        "masks": masks,
                        "iou_predictions": iou_predictions,
                        "low_res_logits": low_res_masks,
                        "high_res_logits": high_res_masks,

                    }
                )
        return outputs

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            # input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        # make it high resolution
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # resize it back to the longest dim (square image)
        masks = F.interpolate(masks, max(original_size),
                              mode="bilinear",
                              align_corners=False
                              )
        # remove padding
        masks = masks[..., : original_size[0], : original_size[1]]
        return masks

    def preprocess(self, x: torch.Tensor, is_input=True) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if is_input:
            if x.shape[0] == 1:
                # Normalize colors map the values in [0,1] to [0,255] for input images and then using
                # original pixel_mean and pixel_std to do normalization
                x = (x * 255.0 - self.pixel_mean) / self.pixel_std
            else:
                # for other 2.5d data, we normalize each input slice
                x = torch.cat(
                    [(x[i].unsqueeze(0) * 255.0 - self.pixel_mean) / self.pixel_std for i in range(x.shape[0])], dim=0)

        # Pad image and make it a square image
        h, w = x.shape[-2:]
        # find the longest dim
        target_length = max(h, w)
        padh = target_length - h
        padw = target_length - w
        x = F.pad(x, (0, padw, 0, padh))
        if is_input:
            # Resize it to self.image_encoder.img_size
            x = F.interpolate(
                x.unsqueeze(0),
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            # Resize it to self.image_encoder.img_size // 4 (for labels). the size is same as low-res logit
            x = F.interpolate(
                x.unsqueeze(0),
                (self.image_encoder.img_size // 4, self.image_encoder.img_size // 4),
                mode="nearest"
            ).squeeze(0)
        return x


def _build_sam2pt5d(
        encoder_in_chans,
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint=None,
        image_size=1024
):
    prompt_embed_dim = 256
    image_size = image_size  # TODO: Shall we try to adapt model to 512x512 ?
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Samm2pt5D(
        image_encoder=ImageEncoderViT(
            in_chans=encoder_in_chans,
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
            num_multimask_outputs=3,  # TODO: only predict one binary mask
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

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        if image_size == 1024:
            # we try to use all pretrained weights
            new_dict = state_dict
        if encoder_in_chans != 3:
            new_dict.pop("image_encoder.patch_embed.proj.weight")
        else:
            new_dict = {}
            for k, v in state_dict.items():
                # skip weights in prompt_encoder and mask_decoder
                if k.startswith("prompt_encoder") or k.startswith("mask_decoder"):
                    continue
                # skip weights in position embedding and learned relative positional embeddings
                elif "pos_embed" in k or "attn.rel_pos" in k:
                    continue
                else:
                    new_dict[k] = v
        sam.load_state_dict(new_dict, strict=False)
        print(f"Load {len(new_dict)} keys from checkpoint {checkpoint}, current model has {len(sam.state_dict())} keys")

        total_params = []
        image_encoder_params = []
        prompt_encoder_params = []
        mask_decoder_params = []
        for name, param in sam.named_parameters():
            n_param = param.numel()
            total_params.append(n_param)
            if name.startswith("image_encoder"):
                image_encoder_params.append(n_param)
            elif name.startswith("prompt_encoder"):
                prompt_encoder_params.append(n_param)
            elif name.startswith("mask_decoder"):
                mask_decoder_params.append(n_param)

        print(f"{sam.__class__.__name__} has {sum(total_params) * 1.e-6:.2f} M params, "
              f"{sum(image_encoder_params) * 1.e-6:.2f} M params in image encoder,"
              f"{sum(prompt_encoder_params) * 1.e-6:.2f} M params in prompt encoder,"
              f"{sum(mask_decoder_params) * 1.e-6:.2f} M params in mask decoder.")

        # comment to unfreeze all encoder layers
        # for name, param in sam.named_parameters():
        #     if name.startswith("image_encoder"):
        #         if image_size == 1024:
        #             if "pos_embed" in name or "patch_embed" in name or "blocks.0" in name:
        #                 # we only retrain layers before blocks.1 in image_encoder
        #                 continue
        #             # if "pos_embed" in name or "patch_embed" in name:
        #             #     # we only retrain pos_embed and patch_embed
        #             #     continue
        #         else:
        #             if "pos_embed" in name or "attn.rel_pos" in name or \
        #                     "patch_embed" in name or "blocks.0" in name or "neck" in name:
        #                 # we only train pos_embed, patch_embed, blocks.0, attn.rel_pos (due res change)
        #                 # and neck (a few conv layers for outputs) in image_encoder
        #                 continue
        #
        #         # we freeze all other layers in image_encoder
        #         param.requires_grad = False

        total_trainable_params = sum(p.numel() if p.requires_grad else 0 for p in sam.parameters())
        print(f"{sam.__class__.__name__} has {total_trainable_params * 1.e-6:.2f} M trainable params.")
    return sam


def build_samm2pt5d_vit_h(checkpoint=None, image_size=1024, encoder_in_chans=3):
    return _build_sam2pt5d(
        encoder_in_chans=encoder_in_chans,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        image_size=image_size,
    )


def build_samm2pt5d_vit_l(checkpoint=None, image_size=1024, encoder_in_chans=3):
    return _build_sam2pt5d(
        encoder_in_chans=encoder_in_chans,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        image_size=image_size,
    )


def build_samm2pt5d_vit_b(checkpoint=None, image_size=1024, encoder_in_chans=3):
    return _build_sam2pt5d(
        encoder_in_chans=encoder_in_chans,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        image_size=image_size,
    )


sam_model_registry = {
    "default": build_samm2pt5d_vit_h,
    "vit_h": build_samm2pt5d_vit_h,
    "vit_l": build_samm2pt5d_vit_l,
    "vit_b": build_samm2pt5d_vit_b,
}

if __name__ == "__main__":
    model = build_samm2pt5d_vit_b()
    model.cuda()
    #
    # dummy_input = [{"image": torch.rand(3, 176, 345).cuda(), "original_size": (176, 345),
    #                 "point_coords": torch.rand(3, 5, 2).cuda(), "point_labels": torch.ones(3, 5).cuda(),
    #                 "labels": torch.ones(3, 1).long().cuda()},
    #                {"image": torch.rand(3, 128, 365).cuda(), "original_size": (128, 365),
    #                 "point_coords": torch.rand(1, 3, 2).cuda(), "point_labels": torch.ones(1, 3).cuda(),
    #                 "labels": torch.ones(1, 1).long().cuda()}
    #                ]
    # # dummy_input = [{"image": torch.rand(3, 176, 345).cuda(), "original_size": (256, 512),
    # #                 "point_coords": torch.rand(3, 5, 2).cuda(), "point_labels": torch.ones(3, 5).cuda()}]
    # outputs = model(dummy_input)

    # test if postprocessing can inverse preprocess
    # path = "/home/pengfeig/Downloads/fffabebf-74fd3a1f-673b6b41-96ec0ac9-2ab69818.jpg"
    # from PIL import Image
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # image = np.array(Image.open(path)).transpose(2, 0, 1)[:, :365, :256].astype(np.float32)
    # plt.imshow(image.transpose(1, 2, 0).astype(np.uint8))
    # plt.show()
    # dummy_tensor = torch.from_numpy(image).cuda()
    # tmp = model.preprocess(dummy_tensor)
    # plt.imshow(tmp.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
    # plt.show()
    # inverse_tensor = model.postprocess_masks(tmp.unsqueeze(0), (365, 256)).squeeze(0)
    # print(torch.sum(torch.abs(inverse_tensor-dummy_tensor)))
    # print("dummy_tensor", torch.min(dummy_tensor), torch.max(dummy_tensor))
    # print("inverse_tensor", torch.min(inverse_tensor), torch.max(inverse_tensor))
    # plt.imshow(inverse_tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
    # plt.show()
    # print()
