# Copyright 2020 - 2023 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch
import copy
from monai import data, transforms
from monai.transforms import (
    LoadImaged,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    Orientationd,
    RandShiftIntensityd,
    RandRotate90d,
    Spacingd,
)


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    train_files, val_files, test_files = split_data(args)

    random_transforms = (
        [
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.10,
            ),
        ]
        if args.data_aug
        else []
    )

    if args.data_aug:
        print("using data augmentation")
    else:
        print("No data augmentation")

    train_transform = transforms.Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
        ]
        + random_transforms
    )

    val_transform = transforms.Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
        ]
    )

    if args.test_mode:
        pass
    else:
        datalist = train_files
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist[:1], transform=train_transform)
        else:
            if args.distributed:
                datalist = data.partition_dataset(
                    data=datalist,
                    shuffle=True,
                    num_partitions=args.world_size,
                    even_divisible=True,
                )[args.rank]

            train_ds = data.CacheDataset(
                data=datalist,
                transform=train_transform,
                cache_rate=1.0,
                num_workers=args.workers,
            )
        train_sampler = None

        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_files = val_files
        if args.distributed:
            val_files = data.partition_dataset(
                data=val_files,
                shuffle=False,
                num_partitions=args.world_size,
                even_divisible=False,
            )[args.rank]
        val_ds = data.CacheDataset(
            data=val_files,
            transform=val_transform,
            cache_rate=1.0,
            num_workers=args.workers,
        )
        val_sampler = None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader


def split_data(args):
    data_dir = args.data_dir
    import json

    with open(args.json_list, "r") as f:
        json_data = json.load(f)

    list_train = []
    list_valid = []
    if "validation" in json_data.keys():
        list_train = json_data["training"]
        list_valid = json_data["validation"]
        list_test = json_data["testing"]
    else:
        for item in json_data["training"]:
            if item["fold"] == args.fold:
                item.pop("fold", None)
                list_valid.append(item)
            else:
                item.pop("fold", None)
                list_train.append(item)
        if "testing" in json_data.keys() and "label" in json_data["testing"][0]:
            list_test = json_data["testing"]
        else:
            list_test = copy.deepcopy(list_valid)
        if args.splitval > 0:
            list_train = sorted(list_train, key=lambda x: x["image"])
            l = int((len(list_train) + len(list_valid)) * args.splitval)
            list_valid = list_train[-l:]
            list_train = list_train[:-l]

    if hasattr(args, "rank") and args.rank == 0:
        print("train files", len(list_train), [os.path.basename(_["image"]).split(".")[0] for _ in list_train])
        print("val files", len(list_valid), [os.path.basename(_["image"]).split(".")[0] for _ in list_valid])
        print("test files", len(list_test), [os.path.basename(_["image"]).split(".")[0] for _ in list_test])

    # training data
    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(data_dir, list_train[_i]["image"])
        str_seg = os.path.join(data_dir, list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    train_files = copy.deepcopy(files)

    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(data_dir, list_valid[_i]["image"])
        str_seg = os.path.join(data_dir, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    val_files = copy.deepcopy(files)

    files = []
    for _i in range(len(list_test)):
        str_img = os.path.join(data_dir, list_test[_i]["image"])
        str_seg = os.path.join(data_dir, list_test[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    test_files = copy.deepcopy(files)
    return train_files, val_files, test_files
