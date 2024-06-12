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
import logging
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime
from typing import Optional, Sequence, Union

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from data.analyzer import (
    RelabelD,
    compute_local_global_label_mappings,
    compute_weights,
    compute_weights_by_ann_voxel,
    compute_dataset_weights,
    get_datalist_with_dataset_name,
    load_class_weights,
    get_datalist_with_dataset_name_and_transform,
    DatasetSelectTansformd,
)
from data.datasets import get_class_names
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
from monai.data import DataLoader, DistributedSampler, DistributedWeightedRandomSampler
from monai.metrics import compute_dice
from monai.utils import set_determinism, RankFilter
from monai.networks.utils import copy_model_state

if __package__ in (None, ""):
    from monai_utils import sliding_window_inference
    from utils import generate_prompt_pairs, get_next_points, generate_prompt_pairs_val, sample_points_patch_val, Point_sampler, none_cat, MERGE_LIST
else:
    from .monai_utils import sliding_window_inference
    from .utils import generate_prompt_pairs, get_next_points, generate_prompt_pairs_val, sample_points_patch_val, Point_sampler,none_cat, MERGE_LIST

import pdb
from functools import partial
from matplotlib import pyplot as plt
from vista3d import vista_model_registry

nib.imageglobals.logger.setLevel(40)
CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"monai_default": {"format": DEFAULT_FMT}},
    "loggers": {
        "monai.apps.auto3dseg.auto_runner": {"handlers": ["file", "console"], "level": "DEBUG", "propagate": False},
        "data.analyzer": {"handlers": ["file", "console"], "level": "INFO", "propagate": False},
        "data.datasets": {"handlers": ["file", "console"], "level": "INFO", "propagate": False},
    },
    "filters": {"rank_filter": {"()": "__main__.RankFilter"}},
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

USE_SV_GT_LIST=[
'TotalSegmentatorV2', 'TotalSegmentatorV2_100', 'Covid19', 'Covid19_100', 
'NLST', 'LIDC', 'LIDC_100', 'StonyBrook-CT', 'StonyBrook-CT_100', 
'TCIA_Colon', 'TCIA_Colon_100'
]                                                                                                                                                                                                                                                                                                                                                                              

def infer_wrapper(inputs, model, **kwargs):
    outputs = model(input_images=inputs,**kwargs)
    return outputs.transpose(1,0)

def loss_wrapper(pred, label, loss_function):
    return loss_function(pred, label)

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
    finetune = parser.get_parsed_content("finetune")
    fold = parser.get_parsed_content("fold")
    num_images_per_batch = parser.get_parsed_content("num_images_per_batch")
    num_epochs = parser.get_parsed_content("num_epochs")
    num_epochs_per_validation = parser.get_parsed_content("num_epochs_per_validation")
    num_sw_batch_size = parser.get_parsed_content("num_sw_batch_size")
    num_patches_per_iter = parser.get_parsed_content("num_patches_per_iter")
    output_classes = parser.get_parsed_content("output_classes")
    overlap_ratio = parser.get_parsed_content("overlap_ratio")
    random_seed = parser.get_parsed_content("random_seed")

    patch_size = parser.get_parsed_content("patch_size")
    input_channels = parser.get_parsed_content("input_channels")
    exclude_background = parser.get_parsed_content("exclude_background", default=True)
    save_last = parser.get_parsed_content("save_last", default=False) 
    save_all = parser.get_parsed_content("save_all", default=False) 
    skip_iter_prob = parser.get_parsed_content("skip_iter_prob")
    reuse_embedding = parser.get_parsed_content("reuse_embedding")
    iter_num = parser.get_parsed_content("iter_num")
    freeze_epoch = parser.get_parsed_content("freeze_epoch", default=-1)
    freeze_head = parser.get_parsed_content("freeze_head", default='auto') 
    use_class_for_point = parser.get_parsed_content("use_class_for_point", default=True)
    max_prompt = parser.get_parsed_content("max_prompt", default=96)
    max_backprompt = parser.get_parsed_content("max_backprompt", default=96)
    max_foreprompt = parser.get_parsed_content("max_foreprompt", default=96)
    drop_label_prob = parser.get_parsed_content("drop_label_prob")
    drop_point_prob = parser.get_parsed_content("drop_point_prob")
    max_point = parser.get_parsed_content("max_point")
    balance_gt = parser.get_parsed_content("balance_gt", default=False)

    weighted_sampling = parser.get_parsed_content("weighted_sampling")
    weighted_loss = parser.get_parsed_content("weighted_loss", default=False)
    train_datasets = parser.get_parsed_content("train_datasets", default=None)
    val_datasets = parser.get_parsed_content("val_datasets", default=None)
    class_names = parser.get_parsed_content("class_names", default=None)
    prefetch_factor = parser.get_parsed_content("prefetch_factor", default=2)
    sample_method = parser.get_parsed_content("sample_method", default='random')

    train_transforms, val_transforms = None, None
    train_transforms = parser.get_parsed_content("transforms_train#transforms", default=None)
    val_transforms = parser.get_parsed_content("transforms_validate#transforms", default=None)

    json_dir = parser.get_parsed_content("json_dir", default='./data/jsons')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        config_yaml = os.path.join(bundle_root, 'configs.yaml')
        ConfigParser.export_config_file(parser.get(), config_yaml, fmt="yaml", default_flow_style=None)
    if random_seed is not None and (isinstance(random_seed, int) or isinstance(random_seed, float)):
        set_determinism(seed=random_seed)

    CONFIG["handlers"]["file"]["filename"] = parser.get_parsed_content("log_output_file")
    logging.config.dictConfig(CONFIG)
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
    logger.debug(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.debug(f"World_size: {world_size}")

    if class_names is None:
        class_names = get_class_names(json_dir=json_dir)

    label_mappings = compute_local_global_label_mappings(json_dir=json_dir)

    image_key, label_key, label_sv_key, pseudo_label_key = parser.get_parsed_content("image_key"), parser.get_parsed_content("label_key"), \
        parser.get_parsed_content("label_sv_key", default=None),parser.get_parsed_content("pseudo_label_key", default=None)
    train_files, _, dataset_specific_transforms, dataset_specific_transforms_val = \
        get_datalist_with_dataset_name_and_transform(datasets=train_datasets,
                                                     fold_idx=fold,
                                                     image_key=image_key,
                                                     label_key=label_key,
                                                     label_sv_key=label_sv_key,
                                                     pseudo_label_key=pseudo_label_key,
                                                     num_patches_per_image=parser.get_parsed_content(
                                                         "num_patches_per_image"),
                                                     patch_size=parser.get_parsed_content("patch_size"),
                                                     json_dir=json_dir
                                                     )
    
    # update supervoxel directory
    json_base = '/data/'
    supervoxel_base = parser.get_parsed_content("supervoxel_base", default='/workspace_infer/supervoxel_sam/')
    for _ in train_files:
        if label_sv_key in _:
            _[label_sv_key] = _[label_sv_key].replace(json_base, supervoxel_base, 1)

    # update pseudo label directory
    json_base = '/data/'
    pl_base = parser.get_parsed_content("pseudolabel_base", default='/workspace_infer/V2_pseudo_12Feb2024/')
    for _ in train_files:
        if pseudo_label_key in _:
            _[pseudo_label_key] = _[pseudo_label_key].replace(json_base, pl_base, 1)


    _, val_files = get_datalist_with_dataset_name(datasets=val_datasets, fold_idx=fold, json_dir=json_dir)
    if torch.cuda.device_count() > 1:
        if len(val_files) < world_size:
            val_files = list(val_files) * math.ceil(float(world_size) / float(len(val_files)))
    logger.debug(f"Train_files: {len(train_files)}")
    logger.debug(f"Val_files: {len(val_files)}")

    if train_transforms is not None:
        # Add dataset-specific transforms to over-sample certain labels
        dataset_select_transform = DatasetSelectTansformd([image_key, label_key], dataset_specific_transforms)
        train_transforms[
            train_transforms.index("Placeholder for dataset-specific transform")] = dataset_select_transform
        logger.debug("using dataset-specific transforms")
        for k, v in dataset_specific_transforms.items():
            logger.debug(k)
            logger.debug(v.transforms)

        label_remapper = RelabelD(label_key, label_mappings=label_mappings, dtype=torch.int32)
        logger.debug("using torch.uint8 type segmentation, support up to 255 classes.")
        train_transforms.append(label_remapper)

        train_transforms = transforms.Compose(train_transforms)

    if val_transforms is not None:
        dataset_select_transform_val = DatasetSelectTansformd([image_key, label_key], dataset_specific_transforms_val)
        val_transforms[
            val_transforms.index("Placeholder for dataset-specific transform")] = dataset_select_transform_val
        logger.debug("using dataset-specific transforms for validation")
        for k, v in dataset_specific_transforms_val.items():
            logger.debug(k)
            try:
                logger.debug(v.transforms)       
            except:
                logger.debug(v) 
        val_transforms = transforms.Compose(val_transforms)


    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=Warning)
        train_ds, val_ds, orig_val_ds = None, None, None
        # for pact 118 use normal dataset to avoid crash.
        train_ds = monai.data.Dataset(
            data=train_files * num_epochs_per_validation,
            transform=train_transforms
        )
        val_ds = monai.data.Dataset(
            data=val_files,
            transform=val_transforms
        )

    train_sampler, val_sampler = None, None
    train_w = None
    class_weight_factor = float(parser.get_parsed_content("class_weight_factor"))
    class_weights = load_class_weights(t=class_weight_factor)  # foreground weights from 1 to 116
    if weighted_loss:
        loss_weights = class_weights.copy()

    if weighted_sampling and train_ds is not None:
        if sample_method == 'random':
            train_w = compute_dataset_weights(train_files, class_weights) * num_epochs_per_validation
            logger.debug('using uniform sample')
        elif sample_method == 'voxel':
            train_w = compute_weights_by_ann_voxel(train_files) * num_epochs_per_validation
            logger.debug('using voxel sample')
        elif sample_method == 'tumor':
            train_w = compute_weights(train_files, class_weights) * num_epochs_per_validation
            logger.debug('using original tumor upsample')

    if torch.cuda.device_count() > 1 and not weighted_sampling:
        if train_ds is not None:
            train_sampler = DistributedSampler(train_ds, shuffle=True)
        if val_ds is not None:
            val_sampler = DistributedSampler(val_ds, shuffle=False, even_divisible=False)

    elif torch.cuda.device_count() == 1 and not weighted_sampling and train_ds is not None:
        train_sampler = RandomSampler(train_ds)
    elif torch.cuda.device_count() == 1 and weighted_sampling and train_ds is not None:
        train_sampler = WeightedRandomSampler(train_w, len(train_files))
    elif torch.cuda.device_count() > 1 and weighted_sampling:
        if train_ds is not None:
            train_sampler = DistributedWeightedRandomSampler(train_ds, train_w)
        if val_ds is not None:
            val_sampler = DistributedSampler(val_ds, shuffle=False, even_divisible=False)

    train_loader = DataLoader(
        train_ds,
        num_workers=parser.get_parsed_content("num_workers"),
        batch_size=num_images_per_batch,
        shuffle=(train_sampler is None),
        persistent_workers=True,
        pin_memory=True,
        sampler=train_sampler,
        prefetch_factor=prefetch_factor if parser.get_parsed_content("num_workers") > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        num_workers=parser.get_parsed_content("num_workers_validation"),
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        prefetch_factor=1 if parser.get_parsed_content("num_workers_validation") > 0 else None,
        persistent_workers=False,
    )

    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}") if world_size > 1 else torch.device("cuda:0")

    model_registry = parser.get_parsed_content("model")
    model = vista_model_registry[model_registry](in_channels=input_channels, image_size=patch_size)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    post_pred = transforms.Compose(
            [transforms.EnsureType(), transforms.AsDiscrete(threshold=0.0, dtype=torch.uint8)]
        )

    optimizer_part = parser.get_parsed_content("optimizer", instantiate=False)
    optimizer = optimizer_part.instantiate(params=model.parameters())

    logger.debug(f"class_weights: {class_weights}")
    logger.debug(f"num_epochs: {num_epochs}")
    logger.debug(f"num_epochs_per_validation: {num_epochs_per_validation}")

    lr_scheduler_part = parser.get_parsed_content("lr_scheduler", instantiate=False)
    lr_scheduler = lr_scheduler_part.instantiate(optimizer=optimizer)

    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    if finetune["activate"] and os.path.isfile(finetune["pretrained_ckpt_name"]):
        logger.debug("Fine-tuning pre-trained checkpoint {:s}".format(finetune["pretrained_ckpt_name"]))
        pretrained_ckpt = torch.load(finetune["pretrained_ckpt_name"], map_location=device)
        copy_model_state(model, pretrained_ckpt, exclude_vars=finetune.get("exclude_vars"))
        del pretrained_ckpt
    else:
        logger.debug("Training from scratch")

    if amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        logger.debug("Amp enabled")

    best_metric = -1
    best_metric_epoch = -1
    idx_iter = 0
    metric_dim = output_classes - 1 if exclude_background else output_classes

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(ckpt_path, "Events"))

        with open(os.path.join(ckpt_path, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

    start_time = time.time()
    metric_class = None
    # To increase speed, the training script is not based on epoch, but based on validation rounds.
    # In each batch, num_images_per_batch=2 whole 3D images are loaded into CPU for data transformation
    # num_patches_per_image=2*num_patches_per_iter is extracted from each 3D image, in each iteration,
    # num_patches_per_iter patches is used for training (real batch size on each GPU).
    num_rounds = int(np.ceil(float(num_epochs) // float(num_epochs_per_validation)))
    if num_rounds == 0:
        raise RuntimeError("num_epochs_per_validation > num_epochs, modify hyper_parameters.yaml")

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        progress_bar = tqdm(
            range(num_rounds), desc=f"{os.path.basename(bundle_root)} - training ...", unit="round"
        )
    for _round in range(num_rounds) if torch.cuda.device_count() > 1 and dist.get_rank() != 0 else progress_bar:
        model.train()
        epoch_loss = 0
        loss_torch = torch.zeros(2, dtype=torch.float, device=device) + 1e-5
        step = 0
        e_time = time.time()
        if torch.cuda.device_count() > 1:
            train_loader.sampler.set_epoch(_round)
        for batch_data in train_loader:
            # for batch_data in cycle_k(train_loader):
            s_time = time.time()
            # if step % (len(train_loader)) == 0:
            if step % (len(train_loader) // num_epochs_per_validation) == 0:
                epoch = _round * num_epochs_per_validation + step // (
                            len(train_loader) // num_epochs_per_validation)
                lr_scheduler.step()
                lr = lr_scheduler.get_last_lr()[0]
                if freeze_epoch > epoch:
                    # if automatic branch is frozen, drop label prompts
                    if freeze_head == 'auto':
                        drop_label_prob_train = 1
                        drop_point_prob_train = 0
                        auto_freeze = True
                        point_freeze = False
                    elif freeze_head == 'point':
                        drop_label_prob_train = 0
                        drop_point_prob_train = 1
                        auto_freeze = False
                        point_freeze = True
                    try:
                        model.module.set_auto_grad(auto_freeze = auto_freeze, point_freeze=point_freeze)
                    except:
                        model.set_auto_grad(auto_freeze = auto_freeze, point_freeze=point_freeze)
                    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                        logger.debug(f'Auto freeze {auto_freeze}, point freeze {point_freeze} at epoch {epoch}!')
                else:
                    drop_label_prob_train = drop_label_prob
                    drop_point_prob_train = drop_point_prob
                    try:
                        model.module.set_auto_grad(auto_freeze=False, point_freeze=False)
                    except:
                        model.set_auto_grad(auto_freeze=False, point_freeze=False)    
                    if torch.cuda.device_count() == 1 or dist.get_rank() == 0: 
                        logger.debug(f'Auto freeze {False}, point freeze {False} at epoch {epoch}!')

                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    logger.debug("----------")
                    logger.debug(f"epoch {epoch}/{num_epochs}")
                    logger.debug(f"Learning rate is set to {lr}")
            step += 1
            inputs_l = batch_data["image"].as_subclass(torch.Tensor)
            if "label" not in batch_data:
                # this will only happen for unlabeled dataset
                batch_data["label"] = batch_data.pop("pseudo_label")
            labels_l = batch_data["label"].as_subclass(torch.Tensor)
            labels_sv_l = batch_data["label_sv"].as_subclass(torch.Tensor) if 'label_sv' in batch_data else None
            labels_p_l = batch_data["pseudo_label"].as_subclass(torch.Tensor) if 'pseudo_label' in batch_data else None

            # if pseudo_label_reliability not exist, default to think it reliable
            if labels_p_l is not None:
                pl_reliability_l = batch_data.get("pseudo_label_reliability", torch.ones(labels_p_l.shape[0], 1))
            else:
                pl_reliability_l = None
            ds_l = batch_data["dataset_name"]
            

            if len(inputs_l) > 1:
                _idx = torch.randperm(inputs_l.shape[0])
                inputs_l = inputs_l[_idx]
                labels_l = labels_l[_idx]
                labels_sv_l = labels_sv_l[_idx] if labels_sv_l is not None else None
                labels_p_l = labels_p_l[_idx] if labels_p_l is not None else None
                pl_reliability_l = pl_reliability_l[_idx] if pl_reliability_l is not None else None
                ds_l = [ds_l[int(_d_i)] for _d_i in _idx]

            for _k in range(inputs_l.shape[0] // num_patches_per_iter):
                inputs = inputs_l[_k * num_patches_per_iter: (_k + 1) * num_patches_per_iter, ...]
                labels = labels_l[_k * num_patches_per_iter: (_k + 1) * num_patches_per_iter, ...]
                labels_sv = labels_sv_l[_k * num_patches_per_iter: (_k + 1) * num_patches_per_iter, ...] if labels_sv_l is not None else None
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels_sv = labels_sv.to(device) if labels_sv is not None else None

                ds_name = ds_l[_k]
                # label_mapping does not contain unlabeled dataset
                try:
                    train_label_set = {_xx[1] for _xx in label_mappings[ds_name]}
                except:
                    train_label_set = {_xx[1] for _xx in label_mappings['TotalSegmentatorV2']}
                train_label_set_pseudo = {_xx[1] for _xx in label_mappings['TotalSegmentatorV2']} | {25, 132}

                pl_reliability = pl_reliability_l[_k * num_patches_per_iter: (_k + 1) * num_patches_per_iter, ...] if pl_reliability_l is not None else None
                labels_p = None
                if pl_reliability is not None and pl_reliability > 0:
                    labels_p = labels_p_l[_k * num_patches_per_iter: (_k + 1) * num_patches_per_iter, ...] if labels_p_l is not None else None
                    labels_p = labels_p.to(device) if labels_p is not None else None
                    # train_label_set = list(set(train_label_set) - set(torch.unique(labels_p).tolist()))
                # # decide if use iterative training and sync across all ranks
                if torch.cuda.device_count() > 1:
                    if dist.get_rank() == 0:
                        skip_iter = (torch.rand(1) < skip_iter_prob).to(dtype=torch.float, device=device)
                    else:
                        skip_iter = torch.empty(1).to(dtype=torch.float, device=device)
                    dist.broadcast(skip_iter, src=0)
                else:
                    skip_iter = (torch.rand(1) < skip_iter_prob).float()
                if skip_iter > 0:
                    # if not using iterative
                    num_iters = 1
                else:
                    # if use iterative training
                    num_iters = max(iter_num, 1)

                point_sampler = None
                point_sampler_pseudo = None
                # for dataset other than totalseg, use pseudolabel for zero-shot. for totalseg, if labels_p exist, use labels_p for zero-shot, 
                # gt for regular sample. If labels_p does not exist, use gt for zero-shot.
                if labels_sv is not None:
                    if labels_p is not None:
                        point_sampler_pseudo = Point_sampler(label=labels_p[0,0], label_sv=labels_sv[0,0], map_shift=512)
                    elif ds_name in USE_SV_GT_LIST:
                        point_sampler = Point_sampler(label=labels[0,0], label_sv=labels_sv[0,0], map_shift=512)

                label_prompt, point, point_label, prompt_class, _ = generate_prompt_pairs(labels,
                                                                        train_label_set,
                                                                        image_size=patch_size,
                                                                        max_point=max_point,
                                                                        max_prompt=max_prompt,
                                                                        max_backprompt=max_backprompt,
                                                                        max_foreprompt=max_foreprompt,
                                                                        drop_label_prob=drop_label_prob_train,
                                                                        drop_point_prob=drop_point_prob_train,
                                                                        include_background=not exclude_background,
                                                                        metric_class=None,
                                                                        ignore_labelset=False,
                                                                        point_sampler=point_sampler)
                label_prompt_pseudo, point_pseudo, point_label_pseudo, prompt_class_pseudo = None, None, None, None

                if labels_p is not None:
                    label_prompt_pseudo, point_pseudo, point_label_pseudo, prompt_class_pseudo, _ = generate_prompt_pairs(labels_p,
                                                                                                        train_label_set_pseudo,
                                                                                                        image_size=patch_size,
                                                                                                        max_point=max_point,
                                                                                                        max_prompt=max_prompt,
                                                                                                        max_backprompt=max_backprompt,
                                                                                                        max_foreprompt=max_foreprompt,
                                                                                                        drop_label_prob=drop_label_prob_train,
                                                                                                        drop_point_prob=drop_point_prob_train,
                                                                                                        include_background=not exclude_background,
                                                                                                        metric_class=None,
                                                                                                        ignore_labelset=False,
                                                                                                        point_sampler=point_sampler_pseudo)
                if point_sampler is not None and prompt_class is not None:
                    # update the labels. The shifted prompt index means zero-shot index. 
                    labels = point_sampler.label.unsqueeze(0).unsqueeze(0)
                    shifted = point_sampler.shifted
                    for i, p in enumerate(prompt_class):
                        if p in list(shifted.keys()):
                            prompt_class[i] = shifted[p.item()]
                del point_sampler
                if point_sampler_pseudo is not None and prompt_class_pseudo is not None:
                    # update the labels. The shifted prompt index means zero-shot index. 
                    labels_p = point_sampler_pseudo.label.unsqueeze(0).unsqueeze(0)
                    shifted = point_sampler_pseudo.shifted
                    for i, p in enumerate(prompt_class_pseudo):
                        if p in list(shifted.keys()):
                            prompt_class_pseudo[i] = shifted[p.item()]
                del point_sampler_pseudo
                torch.cuda.empty_cache()
                # Skip the training if prompts are both None
                skip_update = torch.zeros(1, device=device)
                if label_prompt is None and point is None and label_prompt_pseudo is None and point_pseudo is None:
                    logger.debug(f'Iteration skipped due to None prompts at {ds_name}')
                    skip_update = torch.ones(1, device=device)
                if torch.cuda.device_count() > 1:
                    dist.all_reduce(skip_update, op=dist.ReduceOp.SUM)
                if skip_update[0] > 0:
                    continue  # some rank has no foreground, skip this batch
                # clear image_embedding
                try:
                    model.module.clear_cache()
                except:
                    model.clear_cache()
                for click_indx in range(num_iters):
                    outputs = None
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    labels_p = labels_p.to(device) if labels_p is not None else None
                    # only sinlge point prompt case activate multi-mask output
                    loss_function = partial(loss_wrapper, loss_function=parser.get_parsed_content("loss"))

                    
                    with autocast():
                        outputs = model(input_images=inputs,
                                        point_coords=none_cat(point, point_pseudo),
                                        point_labels=none_cat(point_label, point_label_pseudo),
                                        class_vector=none_cat(label_prompt, label_prompt_pseudo),
                                        prompt_class=none_cat(prompt_class, prompt_class_pseudo),
                                        masks=None,
                                        point_mask=None,
                                        keep_cache=reuse_embedding,
                                        use_cfp=use_class_for_point,
                                        multimask_output=False)
                    # cumulate loss
                    loss, loss_n = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                    ps_start = len(prompt_class) if prompt_class is not None else 0
                    if prompt_class is not None:
                        for idx in range(len(prompt_class)):
                            if exclude_background and prompt_class[idx] == 0:
                                continue  # skip background class
                            loss_n += 1.0
                            gt = labels==prompt_class[idx]
                            if prompt_class[idx].item() in MERGE_LIST.keys():
                                for m in MERGE_LIST[prompt_class[idx].item()]:
                                    gt = torch.logical_or(gt, labels==m)
                            loss_c = loss_function(outputs[[idx]].float(), gt)
                            # debug_pairs(inputs, label=gt, idx=f'{prompt_class[idx].item()}-gt-{step}-{click_indx}')
                            loss += loss_c * loss_weights[prompt_class[idx] - 1] if weighted_loss else loss_c
                    
                    if prompt_class_pseudo is not None:
                        if balance_gt:
                            multiplier = len(prompt_class_pseudo) / len(prompt_class)
                            loss *= multiplier
                        for idx in range(len(prompt_class_pseudo)):
                            if exclude_background and prompt_class_pseudo[idx] == 0:
                                continue  # skip background class
                            loss_n += 1.0
                            gt = labels_p==prompt_class_pseudo[idx]
                            if prompt_class_pseudo[idx].item() in MERGE_LIST.keys():
                                for m in MERGE_LIST[prompt_class_pseudo[idx].item()]:
                                    gt = torch.logical_or(gt, labels_p==m)
                            loss_c = loss_function(outputs[[idx+ps_start]].float(), gt)
                            # debug_pairs(inputs, label=gt, idx=f'{prompt_class_pseudo[idx].item()}-ps-{step}-{click_indx}')
                            loss += loss_c * loss_weights[prompt_class_pseudo[idx] - 1] if weighted_loss else loss_c

                    loss /= max(loss_n, 1.0)
                    print(loss, ds_name, prompt_class, prompt_class_pseudo)
                    if num_iters > 1:
                        if click_indx != num_iters - 1:  # do not sample at last iter
                            outputs.sigmoid_()
                            if prompt_class is not None:
                                point, point_label = get_next_points(outputs[:len(prompt_class)], labels, prompt_class,
                                                                                point, point_label)
                                # debug_next_point(outputs[:len(prompt_class)], point, point_label, None, labels, prompt_class, f'{dist.get_rank()}-{step}-{click_indx}')
                            if prompt_class_pseudo is not None:
                                point_pseudo, point_label_pseudo = get_next_points(outputs[ps_start:], 
                                                                                labels_p, prompt_class_pseudo,
                                                                                point_pseudo, point_label_pseudo)
                                # debug_next_point(outputs[ps_start:], point_pseudo, point_label_pseudo, None, labels_p, prompt_class_pseudo, f'{0}-{step}-{click_indx}-ps')
                            
                            # stop iterative if no new points are added.
                            skip_this_iter = torch.tensor(False, device=device)
                            if prompt_class is not None:
                                if torch.all(point_label[:, -1] == -1) and torch.all(point_label[:, -2] == -1):
                                    skip_this_iter = torch.tensor(True, device=device)
                            if prompt_class_pseudo is not None:
                                if torch.all(point_label_pseudo[:, -1] == -1) and torch.all(point_label_pseudo[:, -2] == -1):
                                    skip_this_iter = torch.tensor(True, device=device)

                            if torch.cuda.device_count() > 1:
                                dist.all_reduce(skip_this_iter, op=dist.ReduceOp.PRODUCT)
                                skip_this_iter = bool(skip_this_iter.item())
                            if skip_this_iter:
                                print(f'iteration end at {click_indx}')
                                logger.info(f'iteration end at {click_indx}')
                                break
                    del outputs
                    torch.cuda.empty_cache()
                        
                    for param in model.parameters():
                        param.grad = None
                    inputs = inputs.to('cpu')
                    labels = labels.to('cpu')
                    labels_p = labels_p.to('cpu') if labels_p is not None else None
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    # clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()

                epoch_loss += loss.item()
                loss_torch[0] += loss.item()
                loss_torch[1] += 1.0
                epoch_len = len(train_loader)  # * num_epochs_per_validation
                idx_iter += 1
                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    logger.debug(
                        f"{time.time() - s_time:.4f} {step}/{epoch_len}, train_loss: {loss.item():.4f}")
                    writer.add_scalar("train/loss", loss.item(), epoch_len * _round + step)

        if torch.cuda.device_count() > 1:
            dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.debug(
                f"{time.time() - e_time:.4f} Epoch {epoch} average loss: {loss_torch_epoch:.4f}, "
                f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )
        try:
            del inputs, labels, inputs_l, labels_l, batch_data, labels_sv_l
        except BaseException:
            pass
        torch.cuda.empty_cache()

        model.eval()
        model_inferer = partial(infer_wrapper, model=model)
        with torch.no_grad():
            # for metric, index 2*c is the dice for class c, and 2*c + 1 is the not-nan counts for class c
            metric = torch.zeros(metric_dim * 2, dtype=torch.float, device=device)
            # One logic about label remapper. In training, labels are mapped to global index. However, there could be local index that are not used.
            # In training sample generation, training pairs will only be sampled from label_set (unless ignore_labelset=True). In validation, the label_prompt 
            # will use global mapping, but the val label is not mapped to global index, so we need the val_orig_set. Notice the compute_dice assume gt label starts 
            # from 0,1,2,3,4,.... If some are index are not used (not defined in label_mapping.json thus label_set does not include them), compute_dice will give wrong
            # number. class_not_sequantial is used to flag this situation.
            for _index, val_data in enumerate(val_loader):
                val_filename = val_data["image"].meta["filename_or_obj"][0]
                val_data["label"] = val_data["label"].as_subclass(torch.Tensor)
                val_data["image"] = val_data["image"].as_subclass(torch.Tensor)
                ds_l = val_data["dataset_name"][0]  # assume batch_size=1
                val_label_set = [0] + [_xx[1] for _xx in label_mappings[ds_l]]
                val_orig_set = [0] + [_xx[0] for _xx in label_mappings[ds_l]]

                if ds_l == "Bone-NIH":
                    val_label_set = val_label_set[:-1]
                    val_orig_set = val_orig_set[:-1]
                    # merge bone lesion 1 and 2
                    val_data["label"][val_data["label"] == 2] = 1

                for _device_in, _device_out in zip([device, device, "cpu"], [device, "cpu", "cpu"]):
                    try:
                        label_prompt, point, point_label, prompt_class = \
                            generate_prompt_pairs_val(val_data["label"].to(_device_in), val_label_set,
                                                        max_ppoint=1,
                                                        device=_device_in)
                        if drop_point_prob_train > 0.99 or (freeze_head=='point' and freeze_epoch > 0) :
                            point = None
                            point_label = None
                        if drop_label_prob_train > 0.99 or (freeze_head=='auto' and freeze_epoch > 0) :
                            label_prompt = None
                        with autocast(enabled=amp):
                            val_outputs = None
                            torch.cuda.empty_cache()
                            val_outputs = sliding_window_inference(
                                inputs=val_data["image"].to(_device_in),
                                roi_size=patch_size,
                                sw_batch_size=num_sw_batch_size,
                                predictor=model_inferer,
                                mode="gaussian",
                                overlap=overlap_ratio,
                                sw_device=device,
                                device=_device_out,
                                point_coords=point,
                                point_labels=point_label,
                                class_vector=label_prompt,
                                prompt_class=prompt_class,
                                labels=val_data["label"].to(_device_in),
                                label_set=val_orig_set,
                                use_cfp=use_class_for_point,
                                brush_radius=None,
                                val_point_sampler=partial(sample_points_patch_val, mapped_label_set=val_label_set, max_ppoint=1, use_center=True))
                        try:
                            val_outputs = post_pred(val_outputs[0, ...])
                        except BaseException:
                            val_outputs = post_pred(val_outputs[0, ...].to("cpu"))
                        finished = True

                    except RuntimeError as e:
                        if not any(x in str(e).lower() for x in ("memory", "cuda", "cudnn")):
                            raise e
                        logger.warning(e)
                        finished = False

                    if finished:
                        break

                if finished:
                    del val_data["image"]
                    value = torch.full((1, metric_dim), float('nan')).to(device)
                    val_outputs = val_outputs[None, ...]
                    value_l = []
                    for i in range(1, len(val_orig_set)):
                        gt = val_data["label"].to(val_outputs.device) == val_orig_set[i]
                        value_l.append(compute_dice(
                            y_pred=val_outputs[:, [i]],
                            y=gt,
                            include_background=False
                        ))
                    value_l = torch.hstack(value_l).to(device)
                    for _v_i, _v_c in enumerate(value_l[0]):
                        value[0, val_label_set[_v_i + 1] - 1] = _v_c
                else:
                    # During training, allow validation OOM for some big data to avoid crush.
                    logger.debug(f'{val_filename} is skipped due to OOM, using NaN dice values')
                    value = torch.full((1, metric_dim), float('nan')).to(device)

                # remove temp variables to save memory.
                val_outputs, val_data = None, None
                torch.cuda.empty_cache()

                logger.debug(f"{_index + 1} / {len(val_loader)} / {val_filename}: {value}")

                for _c in range(metric_dim):
                    val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                    val1 = 1.0 - torch.isnan(value[0, _c]).float()
                    metric[2 * _c] += val0
                    metric[2 * _c + 1] += val1

            if torch.cuda.device_count() > 1:
                dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)

            metric = metric.tolist()
            metric_class = np.zeros(metric_dim)
            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                avg_metric = 0
                valid = 0
                for _c in range(metric_dim):
                    if metric[2 * _c + 1] > 0:
                        v = metric[2 * _c] / metric[2 * _c + 1]
                        avg_metric += v
                        valid += 1
                    else:
                        v = torch.nan
                    metric_class[_c] = v
                    try:
                        writer.add_scalar(f"val_class/acc_{class_names[_c + 1]}", v, epoch)
                        logger.debug(f"Evaluation metric - class {_c + 1} {class_names[_c + 1]}: {v:.4f}")
                    except BaseException:
                        writer.add_scalar(f"val_class/acc_{_c}", v, epoch)
                        logger.debug(f"Evaluation metric - class {_c + 1} : {v:.4f}")

                avg_metric = avg_metric / valid
                logger.debug(f"Avg_metric: {avg_metric}")

                writer.add_scalar("val/acc", avg_metric, epoch)
                if avg_metric > best_metric or save_last:
                    best_metric = avg_metric
                    best_metric_epoch = epoch
                    if save_all:
                        ckpt_name = f"best_metric_model_{epoch}.pt"
                    else:
                        ckpt_name = "best_metric_model.pt"
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), os.path.join(ckpt_path, ckpt_name))
                    else:
                        torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_name))
                    logger.debug("Saved new best metric model")

                    dict_file = {}
                    dict_file["best_avg_dice_score"] = float(best_metric)
                    dict_file["best_avg_dice_score_epoch"] = int(best_metric_epoch)
                    dict_file["best_avg_dice_score_iteration"] = int(idx_iter)
                    with open(os.path.join(ckpt_path, "progress.yaml"), "a") as out_file:
                        yaml.dump([dict_file], stream=out_file)

                logger.debug(
                    "Current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch, avg_metric, best_metric, best_metric_epoch
                    )
                )

                current_time = time.time()
                elapsed_time = (current_time - start_time) / 60.0
                with open(os.path.join(ckpt_path, "accuracy_history.csv"), "a") as f:
                    f.write(
                        "{:d}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.1f}\t{:d}\n".format(
                            epoch, avg_metric, loss_torch_epoch, lr, elapsed_time, idx_iter
                        )
                    )

            if torch.cuda.device_count() > 1:
                dist.barrier()

        torch.cuda.empty_cache()

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        logger.debug(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}.")

        writer.flush()
        writer.close()

        logger.warning(f"{os.path.basename(bundle_root)} - training: finished")

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
