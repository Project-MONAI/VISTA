# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import warnings
from subprocess import Popen

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism
from monai.utils.enums import MetricReduction
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer_2pt5d import run_training
from utils.data_utils import get_loader
from vista_2pt5d.model import sam_model_registry

warnings.filterwarnings("ignore", category=UserWarning, module="monai")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="nibabel")
parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="vista2pt5d", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=1600, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-1024, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=1024, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--fold", default=0, type=int, help="fold")
parser.add_argument(
    "--splitval", default=0, type=float, help="if not zero, split the last portion to validation and validation to test"
)
parser.add_argument("--roi_z_iter", default=9, type=int, help="roi size in z direction")
parser.add_argument("--roi_z_iter_dilation", default=0, type=int, help="dilation size in z direction")
parser.add_argument("--lrschedule", default="No", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--num_patch", default=4, type=int, help="number of patches in each volume")
parser.add_argument("--num_patch_val", default=30, type=int, help="number of patches in each volume during validation")
parser.add_argument("--num_prompt", default=8, type=int, help="number of prompts for each training instance")
parser.add_argument("--clip", default=None, type=float, help="gradient clip")
parser.add_argument("--seed", default=-1, type=int, help="seed")
parser.add_argument("--sam_pretrain_ckpt", type=str, default=None, help="sam_pretrain_ckpt")
parser.add_argument("--sam_base_model", type=str, default="vit_b", help="sam_pretrain_ckpt")
parser.add_argument("--sam_image_size", type=int, default=1024, help="sam input res")
parser.add_argument("--label_prompt", action="store_true", help="using class label prompt in training")
parser.add_argument("--drop_label_prob", default=0.5, type=float, help="prob for dropping label prompt in training")
parser.add_argument(
    "--label_prompt_warm_up_epoch",
    default=20,
    type=int,
    help="before this number of epoch, we will drop label prompt with low prob.",
)
parser.add_argument("--point_prompt", action="store_true", help="using point prompt in training")
parser.add_argument("--drop_point_prob", default=0.5, type=float, help="prob for dropping point prompt in training")
parser.add_argument(
    "--max_points",
    default=8,
    type=int,
    help="max number of point prompts in training for the first ponit prompt generation",
)
parser.add_argument("--points_val_pos", default=1, type=int, help="number of positive point prompts in evaluation")
parser.add_argument("--points_val_neg", default=0, type=int, help="number of negative point prompts in evaluation")
parser.add_argument("--num_iterative_step", default=5, type=int, help="number of iterative step in training")
parser.add_argument("--reuse_img_embedding", action="store_true", help="reuse image embedding in iterative training")
parser.add_argument(
    "--no_more_points_for_cp_only",
    action="store_true",
    help="if no point prompt at the first prompt generation we will not add "
    "more additional pointa during iterative training.",
)
parser.add_argument(
    "--iterative_training_warm_up_epoch",
    default=100,
    type=int,
    help="before this number of epoch, we will not start iterative_training_.",
)
parser.add_argument("--data_aug", action="store_true", help="using data augmentation in training")
parser.add_argument("--pop_pos_embed", action="store_true", help="remove pos embedding when load checkpoint")
parser.add_argument("--pop_point_embed", action="store_true", help="remove point embedding when load checkpoint")
parser.add_argument("--skip_bk", action="store_true", help="skip background (0) during training")
parser.add_argument("--patch_embed_3d", action="store_true", help="using 3d patch embedding layer")

parser.add_argument("--num_classes", default=105, type=int, help="number of output classes")


def start_tb(log_dir):
    cmd = ["tensorboard", "--logdir", log_dir]
    Popen(cmd, stderr=sys.stderr, stdout=sys.stdout, shell=False)


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir

    if args.num_classes == 0:
        warnings.warn("consider setting the correct number of classes")

    # start_tb(args.logdir)
    if args.seed > -1:
        set_determinism(seed=args.seed)
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    model = sam_model_registry[args.sam_base_model](
        args.sam_pretrain_ckpt,
        image_size=args.sam_image_size,
        encoder_in_chans=args.roi_z_iter * 3,
        patch_embed_3d=args.patch_embed_3d,
    )

    dice_loss = DiceCELoss(sigmoid=True)

    post_label = AsDiscrete(to_onehot=args.num_classes)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters count", pytorch_total_params * 1.0e-6, "M")

    best_acc = 0
    start_epoch = 0
    optimizer_state = None

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k] = v
        if args.pop_pos_embed:
            print("pop_pos_embed")
            new_state_dict.pop("image_encoder.patch_embed.proj.weight")
            new_state_dict.pop("image_encoder.patch_embed.proj.bias")
            model.load_state_dict(new_state_dict, strict=False)
        elif args.pop_point_embed:
            print("pop_point_embed")
            new_state_dict.pop("prompt_encoder.point_embeddings.0.weight")
            new_state_dict.pop("prompt_encoder.point_embeddings.1.weight")
            new_state_dict.pop("prompt_encoder.point_embeddings.2.weight")
            new_state_dict.pop("prompt_encoder.point_embeddings.3.weight")
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=True)
        if args.resume_ckpt:
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            if "best_acc" in checkpoint:
                best_acc = checkpoint["best_acc"]
            if "optimizer" in checkpoint:
                optimizer_state = checkpoint["optimizer"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        # override lr by the given value
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.optim_lr

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
