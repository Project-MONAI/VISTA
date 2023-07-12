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

import numpy as np
import scipy.ndimage as ndimage
import torch

SAM_IMAGE_SIZE = 1024

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def prepare_sam_val_input(inputs, class_prompts, point_prompts, start_idx, previous_pred=None, cachedEmbedding=None):
    # Don't exclude background in val but will ignore it in metric calculation
    H,W = inputs.shape[1:]
    foreground_all = point_prompts["foreground"]
    background_all = point_prompts["background"]

    class_list = [[i+1] for i in class_prompts]
    unique_labels = torch.tensor(class_list).long().cuda()

    volume_point_coords = [cp for cp in foreground_all]
    volume_point_labels = [1]*len(foreground_all)

    for cp in background_all:
        volume_point_coords.append(cp)
        volume_point_labels.append(0)

    # volume_point_coords = foreground_all + background_all
    # volume_point_labels = [1] * len(foreground_all) + [0] * len(background_all)

    point_coords = [[]]
    point_labels = [[]]
    for idx, cp in enumerate(volume_point_coords):
        if cp[2]+4 == start_idx:
            new_H = cp[0] * (SAM_IMAGE_SIZE / H)
            new_W = cp[1] * (SAM_IMAGE_SIZE / W)
            point_coords[0].append([new_H, new_W])
            point_labels[0].append(volume_point_labels[idx])


    if len(point_coords[0]) == 0:
        point_coords = None 
        point_labels = None

    prepared_input = [{"image": inputs, "original_size": tuple(inputs.shape[1:])}]
        
    if len(class_prompts) == 0:
        class_enabled = False
    else:
        class_enabled = True
    if class_enabled:
        prepared_input[0].update(
            {"labels": unique_labels})

    if point_coords:
        point_coords = torch.tensor(point_coords).long().cuda()
        point_labels = torch.tensor(point_labels).long().cuda()


        prepared_input[0].update(
            {"point_coords": point_coords, "point_labels": point_labels})
        
    return prepared_input, unique_labels
