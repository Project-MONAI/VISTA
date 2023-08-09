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

import contextlib
import ctypes
import io
import logging
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import timedelta
import monai
from monai import transforms
from monai.apps import download_url
from monai.apps.auto3dseg.auto_runner import logger
from monai.apps.utils import DEFAULT_FMT
from monai.auto3dseg.utils import datafold_read
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import DataLoader, partition_dataset
from .monai_utils import sliding_window_inference
from monai.metrics import compute_dice
from monai.utils import set_determinism
import copy
import pdb
from functools import partial
from .utils import generate_prompt_pairs_val, get_next_points, debug_next_point, point_based_window_inferer, pad_previous_mask
from matplotlib import pyplot as plt
from segment_anything3d import sam_model_registry

CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"monai_default": {"format": DEFAULT_FMT}},
    "loggers": {
        "monai.apps.auto3dseg.auto_runner": {"handlers": ["file", "console"], "level": "DEBUG", "propagate": False}
    },
    "filters": {"rank_filter": {"{}": "__main__.RankFilter"}},
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "runner.log",
            "mode": "a",  # append or overwrite
            "level": "DEBUG",
            "formatter": "monai_default",
            "filters": ["rank_filter"],
        },
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "monai_default",
            "filters": ["rank_filter"],
        },
    },
}

def infer_wrapper(inputs, model, **kwargs):
    outputs = model(input_images=inputs, **kwargs)
    return outputs.transpose(1,0)


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    # Initialize distributed and scale parameters based on GPU memory
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=3600))
        world_size = dist.get_world_size()
        dist.barrier()
    else:
        world_size = 1

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if isinstance(config_file, str) and "," in config_file:
        config_file = config_file.split(",")

    _args = _update_args(config_file=config_file, **override)
    config_file_ = _pop_args(_args, "config_file")[0]

    parser = ConfigParser()
    parser.read_config(config_file_)
    parser.update(pairs=_args)

    amp = parser.get_parsed_content("amp")
    bundle_root = parser.get_parsed_content("bundle_root")
    ckpt_path = parser.get_parsed_content("ckpt_path")
    data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
    data_list_file_path = parser.get_parsed_content("data_list_file_path")
    finetune = parser.get_parsed_content("finetune")
    fold = parser.get_parsed_content("fold")
    num_images_per_batch = parser.get_parsed_content("num_images_per_batch")
    num_epochs = parser.get_parsed_content("num_epochs")
    num_epochs_per_validation = parser.get_parsed_content("num_epochs_per_validation")
    num_sw_batch_size = parser.get_parsed_content("num_sw_batch_size")
    num_patches_per_iter = parser.get_parsed_content("num_patches_per_iter")
    output_classes = parser.get_parsed_content("output_classes")
    overlap_ratio = parser.get_parsed_content("overlap_ratio")
    overlap_ratio_final = parser.get_parsed_content("overlap_ratio_final")
    patch_size_valid = parser.get_parsed_content("patch_size_valid")
    random_seed = parser.get_parsed_content("random_seed")
    sw_input_on_cpu = parser.get_parsed_content("sw_input_on_cpu")
    softmax = parser.get_parsed_content("softmax")
    valid_at_orig_resolution_at_last = parser.get_parsed_content("valid_at_orig_resolution_at_last")
    valid_at_orig_resolution_only = parser.get_parsed_content("valid_at_orig_resolution_only")
    use_pretrain = parser.get_parsed_content("use_pretrain")
    pretrained_path = parser.get_parsed_content("pretrained_path")

    patch_size = parser.get_parsed_content("patch_size")
    model_registry = parser.get_parsed_content("model")
    input_channels = parser.get_parsed_content("input_channels")
    label_set = parser.get_parsed_content("label_set")
    if label_set is None:
        label_set = np.arange(output_classes).tolist()
    max_point = parser.get_parsed_content("max_point")

    val_transforms = parser.get_parsed_content("transforms_validate")

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)

    if random_seed is not None and (isinstance(random_seed, int) or isinstance(random_seed, float)):
        set_determinism(seed=random_seed)

    CONFIG["handlers"]["file"]["filename"] = parser.get_parsed_content("validate#log_output_file")
    logging.config.dictConfig(CONFIG)
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
    logger.debug(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.debug(f"World_size: {world_size}")

    train_files, val_files = datafold_read(datalist=data_list_file_path, basedir=data_file_base_dir, fold=fold)

    random.shuffle(train_files)
    if torch.cuda.device_count() > 1:
        if len(val_files) < world_size:
            val_files = val_files * math.ceil(float(world_size) / float(len(val_files)))

        val_files = partition_dataset(data=val_files, shuffle=False, num_partitions=world_size, even_divisible=False)[
            dist.get_rank()
        ]
    logger.debug(f"Val_files: {len(val_files)}")


    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, num_workers=parser.get_parsed_content("num_workers_validation"), batch_size=1, shuffle=False)

    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}") if world_size > 1 else torch.device("cuda:0")

    model = sam_model_registry[model_registry](in_channels=input_channels, image_size=patch_size)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if softmax:
        post_pred = transforms.Compose([transforms.EnsureType(), transforms.AsDiscrete(argmax=True)])
    else:
        post_pred = transforms.Compose(
            [transforms.EnsureType(), transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)]
        )

    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    if finetune["activate"] and os.path.isfile(finetune["pretrained_ckpt_name"]):
        logger.debug("Fine-tuning pre-trained checkpoint {:s}".format(finetune["pretrained_ckpt_name"]))
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(finetune["pretrained_ckpt_name"], map_location=device), strict=False)
        else:
            model.load_state_dict(torch.load(finetune["pretrained_ckpt_name"], map_location=device), strict=False)
    else:
        if not use_pretrain:
            logger.debug("Training from scratch")

    if amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            logger.debug("Amp enabled")

    best_metric = -1
    best_metric_epoch = -1
    idx_iter = 0
    metric_dim = output_classes - 1 if softmax else output_classes
    val_devices_input = {}
    val_devices_output = {}

    start_time = time.time()

    model.eval()
    metric_list = {}
    # label only + label + 1 point + iterative point
    label_start = True
    point_start_num = 0
    max_point = 3
    
    max_iters = max_point + int(label_start) - point_start_num
    model_inferer = partial(infer_wrapper, model=model)      
    with torch.no_grad():
        metric = torch.zeros(max_iters, metric_dim * 2, dtype=torch.float, device=device)
        _index = 0
        for val_data in val_loader:
            try:
                val_filename = val_data["image_meta_dict"]["filename_or_obj"][0]
            except BaseException:
                val_filename = val_data["image"].meta["filename_or_obj"][0]
            _index += 1
            for idx in range(max_iters):
                # generate points
                if idx == 0:
                    label_prompt, point, point_label, prompt_class, point_mask = \
                        generate_prompt_pairs_val(val_data["label"], label_set, 
                                                image_size=patch_size,
                                                max_point=max(point_start_num, 1),
                                                device='cpu')
                    prev_mask = None                   
                    sliding_window_inferer = sliding_window_inference    
                         
                else:
                    sliding_window_inferer = point_based_window_inferer
                with autocast(enabled=amp):
                    val_outputs = sliding_window_inferer(	
                                            inputs=val_data["image"].to(device),	
                                            roi_size=patch_size_valid,	
                                            sw_batch_size=num_sw_batch_size,	
                                            predictor=model_inferer,	
                                            mode="gaussian",	
                                            overlap=overlap_ratio_final,	
                                            sw_device=device,	
                                            device=device,
                                            point_coords=point.to(device) if point is not None else None,
                                            point_labels=point_label.to(device) if point_label is not None else None,
                                            class_vector=label_prompt.to(device) if label_prompt is not None else None,
                                            masks=prev_mask.to(device) if prev_mask is not None else None,
                                            point_mask=None)
                prev_mask = val_outputs.as_tensor() if isinstance(val_outputs, monai.data.MetaTensor) else val_outputs
                try:
                    val_outputs = post_pred(val_outputs[0, ...])
                except BaseException:
                    val_outputs = post_pred(val_outputs[0, ...].to("cpu"))
                val_outputs = val_outputs[None, ...]
                value = compute_dice(
                    y_pred=val_outputs,
                    y=val_data["label"].to(val_outputs.device),
                    include_background=not softmax,
                    num_classes=output_classes,
                ).to(device)

                # move all to cpu to avoid potential out memory in invert transform
                torch.cuda.empty_cache()

                point, point_label, point_mask = get_next_points((prev_mask.float().sigmoid().transpose(1,0).cpu().clone() > 0.5).float(), val_data["label"], prompt_class, 
                                                                 point, point_label, image_size=patch_size, previous_mask=point_mask)    

                logger.debug(
                    f"Validation Dice score : {idx} / {_index + 1} / {len(val_loader)}/ {val_filename}: {value}"
                )

                for _c in range(metric_dim):
                    val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                    val1 = 1.0 - torch.isnan(value[0, _c]).float()
                    metric[idx, 2 * _c] += val0
                    metric[idx, 2 * _c + 1] += val1

        if torch.cuda.device_count() > 1:
            dist.barrier()
            dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)

        # metric = metric.tolist()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            for _i in range(max_iters):
                for _c in range(metric_dim):
                    logger.debug(f"Evaluation metric at {_i} - class {_c + 1}: {metric[_i, 2 * _c] / metric[_i, 2 * _c + 1]}")

            for _i in range(max_iters):
                avg_metric = 0
                for _c in range(metric_dim):
                    avg_metric += metric[_i, 2 * _c] / metric[_i, 2 * _c + 1]
                avg_metric = avg_metric / float(metric_dim)
                logger.debug(f"Avg_metric at {_i}: {avg_metric}")
                logger.debug(
                    "Current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        0, avg_metric, best_metric, best_metric_epoch
                    )
                )
        if torch.cuda.device_count() > 1:
            dist.barrier()
    torch.cuda.empty_cache()
    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()