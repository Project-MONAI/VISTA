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

import os
import warnings
from collections.abc import Hashable, Mapping, Sequence
from time import time
import pdb
import monai.transforms as mt
import numpy as np
import torch
from monai import transforms
from monai.apps import get_logger
from monai.apps.auto3dseg import DataAnalyzer
from monai.auto3dseg.utils import datafold_read
from monai.bundle import ConfigParser
from monai.config import DtypeLike, KeysCollection, NdarrayOrTensor
from monai.transforms import MapTransform
from monai.utils import (
    TransformBackends,
    WorkflowProfiler,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    get_equivalent_dtype,
    look_up_option,
)

from .datasets import all_base_dirs, get_json_files_k_folds, get_label_dict_k_folds, valid_base_dirs, pseudo_base_dirs

logger = get_logger(__name__)


class MapLabelValue:
    """
    Utility to map label values to another set of values.
    For example, map [3, 2, 1] to [0, 1, 2], [1, 2, 3] -> [0.5, 1.5, 2.5], ["label3", "label2", "label1"] -> [0, 1, 2],
    [3.5, 2.5, 1.5] -> ["label0", "label1", "label2"], etc.
    The label data must be numpy array or array-like data and the output data will be numpy array.

    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, orig_labels: Sequence, target_labels: Sequence, dtype: DtypeLike = np.float32) -> None:
        """
        Args:
            orig_labels: original labels that map to others.
            target_labels: expected label values, 1: 1 map to the `orig_labels`.
            dtype: convert the output data to dtype, default to float32.
                if dtype is from PyTorch, the transform will use the pytorch backend, else with numpy backend.

        """
        if len(orig_labels) != len(target_labels):
            raise ValueError("orig_labels and target_labels must have the same length.")

        self.orig_labels = orig_labels
        self.target_labels = target_labels
        self.pair = tuple((o, t) for o, t in zip(self.orig_labels, self.target_labels) if o != t)
        type_dtype = type(dtype)
        if getattr(type_dtype, "__module__", "") == "torch":
            self.use_numpy = False
            self.dtype = get_equivalent_dtype(dtype, data_type=torch.Tensor)
        else:
            self.use_numpy = True
            self.dtype = get_equivalent_dtype(dtype, data_type=np.ndarray)

    def __call__(self, img: NdarrayOrTensor):
        if self.use_numpy:
            img_np, *_ = convert_data_type(img, np.ndarray)
            _out_shape = img_np.shape
            img_flat = img_np.flatten()
            try:
                out_flat = img_flat.astype(self.dtype)
            except ValueError:
                # can't copy unchanged labels as the expected dtype is not supported, must map all the label values
                out_flat = np.zeros(shape=img_flat.shape, dtype=self.dtype)
            for o, t in self.pair:
                out_flat[img_flat == o] = t
            out_t = out_flat.reshape(_out_shape)
        else:
            img_t, *_ = convert_data_type(img, torch.Tensor)
            out_t = img_t.detach().clone().to(self.dtype)  # type: ignore
            for o, t in self.pair:
                out_t[img_t == o] = t
        out, *_ = convert_to_dst_type(src=out_t, dst=img, dtype=self.dtype)
        return out

class DatasetSelectTansformd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            dataset_transforms,
            dataset_key: str = "dataset_name",
            allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            dataset_transforms: a dictionary specifies the transform for corresponding dataset:
                key: dataset name, value: list of data transforms.
            dataset_key: key to get the dataset name from the data dictionary, default to "dataset_name".
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.dataset_transforms = dataset_transforms
        self.dataset_key = dataset_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        dataset_name = d[self.dataset_key]
        _m = self.dataset_transforms[dataset_name]
        if _m is None:
            return d
        return _m(d)

class RelabelD(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            label_mappings: dict[str, list[tuple[int, int]]],
            dtype: DtypeLike = np.int16,
            dataset_key: str = "dataset_name",
            allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            label_mappings: a dictionary specifies how local dataset class indices are mapped to the
                global class indices, format:
                key: dataset name, value: list of (local label, global label) pairs
            dtype: convert the output data to dtype, default to float32.
            dataset_key: key to get the dataset name from the data dictionary, default to "dataset_name".
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.mappers = {}
        self.dataset_key = dataset_key
        for name, mapping in label_mappings.items():
            self.mappers[name] = MapLabelValue(
                orig_labels=[pair[0] for pair in mapping], target_labels=[pair[1] for pair in mapping], dtype=dtype
            )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        dataset_name = d[self.dataset_key]
        try:
            _m = look_up_option(dataset_name, self.mappers)
            for key in self.key_iterator(d):
                d[key] = _m(d[key])
            return d
        except:
            return d   


Relabeld = RelabelDict = RelabelD


def compute_local_global_label_mappings(json_dir=None, base_dirs=None, output_mapping_json=None):
    """
    Given dataset JSON files specified by json_dirs, base_dirs, compute the local label to global label mapping.
    If the output_mapping_json exists, load the label mappings from the file, else compute the label mappings and
    save to the output_mapping_json file.
    """
    if json_dir is None:
        label_mappings_config = os.path.join(os.path.dirname(__file__), "jsons", "label_mappings.json")
    else:
        label_mappings_config = os.path.join(json_dir, "label_mappings.json")

    output_json = label_mappings_config if output_mapping_json is None else output_mapping_json
    if os.path.exists(output_json):
        logger.warning(f"using global label mapping file {output_json}")
        label_mappings = dict(ConfigParser.load_config_file(output_json))
        _label = {}
        for _ in label_mappings.keys():
            _label[_] = label_mappings[_]
            _label[_+'_100'] = label_mappings[_]
        return _label
    # get the global label map dictionary
    global_label_dict = get_label_dict_k_folds(json_dir=json_dir, base_dirs=base_dirs)
    label_mappings = {}  # key: dataset name, value: list of (local label, global label) pairs
    json_files = get_json_files_k_folds(json_dir=json_dir, base_dirs=base_dirs)
    for k, j in json_files.items():
        local_label_dict = ConfigParser.load_config_file(j)["label_dict"]
        label_mappings[k] = [  # integer to integer, local index -> global index
            (int(cls_idx), int(global_label_dict[cls_name])) for cls_idx, cls_name in local_label_dict.items()
        ]
        # label_mappings[k].append((0, 0))
    if {pair[1] for xx in label_mappings.values() for pair in xx} != set(global_label_dict.values()):
        warnings.warn("!!!! union of label mapping targets != global label set, not 1-1 mappings?", stacklevel=2)
    logger.warning("writing label mappings to file: ", output_json)
    ConfigParser.export_config_file(label_mappings, output_json, indent=4, separators=(",", ":"))
    return label_mappings


def get_datalist_with_dataset_name(
        datasets=None, fold_idx=-1, key="training", json_dir=None, base_dirs=None, output_mapping_json=None,
        pseudo_label_key=None
):
    """
    when `datasets` is None, it returns a list of all data from all datasets.
    when `datasets` is a list of dataset names, it returns a list of all data from the specified datasets.

    train_list's item format::

        {"image": image_file_path, "label": label_file_path, "dataset_name": dataset_name, "fold": fold_id}

    """
    if base_dirs is None:
        base_dirs = valid_base_dirs  # all_base_dirs is the broader set
    # get the list of training/validation files (absolute path)
    json_files = get_json_files_k_folds(json_dir=json_dir, base_dirs=base_dirs)
    if datasets is None:
        loading_dict = json_files.copy()
    else:
        loading_dict = {k: look_up_option(k, json_files) for k in ensure_tuple(datasets)}
    train_list, val_list = [], []
    for k, j in loading_dict.items():
        t, v = datafold_read(j, basedir=base_dirs[k], fold=fold_idx, key=key)
        for item in t:
            item["dataset_name"] = k
            if pseudo_label_key is not None and k in pseudo_base_dirs and pseudo_label_key in item:
                item[pseudo_label_key] = item[pseudo_label_key].replace(base_dirs[k], pseudo_base_dirs[k])
        train_list += t
        for item in v:
            item["dataset_name"] = k
            if pseudo_label_key is not None and k in pseudo_base_dirs and pseudo_label_key in item:
                item[pseudo_label_key] = item[pseudo_label_key].replace(base_dirs[k], pseudo_base_dirs[k])
        val_list += v
    logger.warning(f"data list from datasets={datasets} fold={fold_idx}: train={len(train_list)}, val={len(val_list)}")
    return ensure_tuple(train_list), ensure_tuple(val_list)


def get_datalist_with_dataset_name_and_transform(
        image_key, label_key, label_sv_key, pseudo_label_key, num_patches_per_image, patch_size,
        datasets=None, fold_idx=-1, key="training", json_dir=None, base_dirs=None, output_mapping_json=None
):
    """
    when `datasets` is None, it returns a list of all data from all datasets.
    when `datasets` is a list of dataset names, it returns a list of all data from the specified datasets.

    Return file lists and specific transforms for each dataset.

    """

    if base_dirs is None:
        base_dirs = valid_base_dirs  # all_base_dirs is the broader set
        _label = {}
        for _ in base_dirs.keys():
            _label[_] = base_dirs[_]
            _label[_+'_100'] = base_dirs[_]
        base_dirs = _label
    train_list, val_list = get_datalist_with_dataset_name(datasets=datasets, fold_idx=fold_idx, key=key,
                                                          json_dir=json_dir, base_dirs=base_dirs,
                                                          output_mapping_json=output_mapping_json,
                                                          pseudo_label_key=pseudo_label_key)
    # get the list of training/validation files (absolute path)
    json_files = get_json_files_k_folds(json_dir=json_dir, base_dirs=base_dirs)
    if datasets is None:
        loading_dict = json_files.copy()
    else:
        loading_dict = {k: look_up_option(k, json_files) for k in ensure_tuple(datasets)}

    dataset_transforms = {}
    dataset_transforms_val = {}
    for k, j in loading_dict.items():
        parser = ConfigParser()
        parser.read_config(j)
        parser.update(
            pairs={"image_key": image_key,
                   "label_key": label_key,
                   "label_sv_key": label_sv_key,
                   "pseudo_label_key": pseudo_label_key,
                   "num_patches_per_image": num_patches_per_image,
                   "patch_size": patch_size})
        transform = parser.get_parsed_content("training_transform")
        dataset_transforms[k] = transforms.Compose(transform)
        transform_val = parser.get_parsed_content("validation_transform", default=None)
        dataset_transforms_val[k] = transforms.Compose(transform_val) if transform_val is not None else None
    return ensure_tuple(train_list), ensure_tuple(val_list), dataset_transforms, dataset_transforms_val


def get_data_stats_transforms(label_mappings=None):
    """transforms used to compute image and label stats"""
    if label_mappings is None:
        logger.warning("using global default label_mappings")
        label_mappings = compute_local_global_label_mappings()
    mapping_transform = [
        mt.LoadImaged(("image", "label"), image_only=True, dtype=None, ensure_channel_first=True),
        mt.EnsureTyped(keys="image", data_type="tensor", dtype=torch.float),
        mt.Orientationd(keys=("image", "label"), axcodes="RAS"),
        #         mt.Spacingd(keys=("image", "label"), pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        RelabelD("label", label_mappings=label_mappings, dtype=torch.int16),
        mt.EnsureTyped(keys="label", data_type="tensor", dtype=torch.int16),
    ]
    return mapping_transform


def get_label_stats_transforms(label_mappings=None):
    """
    transforms used to compute label frequencies

    ```
    import monai
    from data.analyzer import get_label_stats_transforms

    xforms = get_label_stats_transforms()
    xforms = monai.transforms.Compose(xforms)

    out = xforms({'label': "totalsegmentator-example/labels/s0124.nii.gz", "dataset_name": "TotalSegmentator"})
    monai.visualize.matshow3d(out['label'], show=True)

    ```
    """
    if label_mappings is None:
        logger.warning("using global default label_mappings")
        label_mappings = compute_local_global_label_mappings()
    label_transform = [
        mt.LoadImaged("label", image_only=True, dtype=None, ensure_channel_first=True),
        mt.Orientationd("label", axcodes="RAS"),
        mt.Spacingd("label", pixdim=(1.5, 1.5, 1.5), mode="nearest"),
        RelabelD("label", label_mappings=label_mappings, dtype=torch.int16),
        mt.EnsureTyped("label", data_type="tensor", dtype=torch.float),
    ]
    return label_transform


def compute_image_label_stats(train_list=None):
    if train_list is None:
        train_list = get_datalist_with_dataset_name()[0]
    analyzer = DataAnalyzer({"training": train_list}, "")
    analyzer.get_all_case_stats(transform_list=get_data_stats_transforms())


def compute_label_histogram_stats(train_list=None):
    """compute all case label stats, this is used to generate class frequencies"""
    if train_list is None:
        train_list = get_datalist_with_dataset_name()[0]
    analyzer = DataAnalyzer(
        {"training": train_list},
        output_path="./hist.yaml",
        image_key="label",
        label_key=None,
        hist_bins=[200],
        hist_range=[[-0.5, 199.5]],
        histogram_only=True,
    )
    analyzer.get_all_case_stats(transform_list=get_label_stats_transforms())
    cfg = ConfigParser.load_config_file("./hist_by_case.yaml")
    n_samples = len(cfg["stats_by_cases"])
    name_id = {cfg["stats_by_cases"][i]["image_filepath"]: i for i in range(n_samples)}
    ConfigParser.export_config_file(name_id, "./hists.yaml", fmt="yaml")


def load_class_weights(t=0.001):
    """compute class-wise weight, higher the more likely to be sampled"""
    cfg = ConfigParser.load_config_file("./hist.yaml")
    counts = np.asarray(cfg["stats_summary"]["image_histogram"]["histogram"][0]["counts"])
    counts = counts / counts.sum()
    counts = counts[1:]  # ignore background
    counts = counts[counts > 0]
    weights = np.maximum(1.0, np.sqrt(t / counts))
    return weights


def compute_weights(datalist, class_weights):
    """based on class-wise weight, assign a weight to each training sample"""
    cfg = ConfigParser.load_config_file("./hist_by_case.yaml")
    n_samples = len(cfg["stats_by_cases"])
    name_id = {cfg["stats_by_cases"][i]["image_filepath"]: i for i in range(n_samples)}
    w = []
    for item in datalist:
        hist = np.asarray(cfg["stats_by_cases"][name_id[item["label"]]]["image_histogram"][0]["counts"].copy())
        hist = hist[1: (len(class_weights) + 1)]  # ignore background
        ws = np.array(class_weights)
        fg_weights = ws[hist > 0]
        # this volume has too many labels, treating it as less freq
        fg_w = fg_weights.max() if len(fg_weights) <= 10 else fg_weights.min()
        w.append(fg_w)
        item["w"] = fg_w
    return w

def compute_dataset_weights(datalist, class_weights):
    """based on class-wise weight, assign a weight to each training sample"""
    cfg = ConfigParser.load_config_file("./dataset_weights.yaml")
    w = []
    for item in datalist:
        fg_w = cfg[item['dataset_name']]
        w.append(fg_w)
        item["w"] = fg_w
    return w

def compute_weights_by_ann_voxel(datalist):
    """based on the number of annotated, assign a weight to each training sample"""
    cfg = ConfigParser.load_config_file("./hist_by_case.yaml")
    n_samples = len(cfg["stats_by_cases"])
    name_id = {cfg["stats_by_cases"][i]["image_filepath"]: i for i in range(n_samples)}

    counts = np.asarray(cfg["stats_summary"]["image_histogram"]["histogram"][0]["counts"])[1:] # ignore background
    total_ann_voxel = counts.sum()

    w = []
    for item in datalist:
        hist = np.asarray(cfg["stats_by_cases"][name_id[item["label"]]]["image_histogram"][0]["counts"].copy())
        hist = hist[1: ].sum()  # ignore background
        fg_weights = hist/total_ann_voxel * 1e7
        # this volume has too many labels, treating it as less freq
        # fg_w = fg_weights.max() if len(fg_weights) <= 10 else fg_weights.min()
        fg_w = fg_weights
        w.append(fg_w)
        item["w"] = fg_w
    return w

def quick_test(idx=5):
    """test timing for data loading, `python -m data.analyzer quick_test --idx=1700`"""
    idx = int(idx)
    b = time()
    xform = mt.Compose(get_data_stats_transforms())
    with WorkflowProfiler() as wp:
        for item in get_datalist_with_dataset_name()[0][idx - 3:idx + 3]:
            a = time()
            output = xform(item)["label"]
            print(output.unique(), output.shape, output.meta['filename_or_obj'], time() - a)
    print("total time", time() - b)
    print(wp.get_times_summary_pd())


if __name__ == "__main__":
    """
    run label histogram analyzer

        - `python -m data.analyzer compute_label_histogram_stats`
    """
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()

