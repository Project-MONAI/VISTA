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
from collections.abc import Hashable, Mapping
from time import time

import monai.transforms as mt
import numpy as np
import torch
from monai.apps import get_logger
from monai.apps.auto3dseg import DataAnalyzer
from monai.auto3dseg.utils import datafold_read
from monai.bundle import ConfigParser
from monai.config import DtypeLike, KeysCollection, NdarrayOrTensor
from monai.transforms import MapLabelValue, MapTransform
from monai.utils import WorkflowProfiler, ensure_tuple, look_up_option

from .datasets import all_base_dirs, get_json_files_k_folds, get_label_dict_k_folds, all_base_dirs

logger = get_logger(__name__)

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
                set this argument to "{}" to disable relabeling.
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
        dataset_name = d.get(self.dataset_key, "default")
        _m = look_up_option(dataset_name, self.mappers, default=None)
        if not _m:
            return d
        for key in self.key_iterator(d):
            d[key] = _m(d[key])
        return d


Relabeld = RelabelDict = RelabelD
_cache_map = None


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
    if os.path.exists(label_mappings_config):
        logger.warning(f"using global label mapping file {label_mappings_config}")
        label_mappings = dict(ConfigParser.load_config_file(label_mappings_config))
        _label = {}
        for _ in label_mappings.keys():
            _label[_] = label_mappings[_]
            _label[_+'_100'] = label_mappings[_]
        return _label
    else:
        raise NotImplementedError


def get_datalist_with_dataset_name(
    datasets=None, fold_idx=-1, key="training", json_dir=None, base_dirs=None, output_mapping_json=None
):
    """
    when `datasets` is None, it returns a list of all data from all datasets.
    when `datasets` is a list of dataset names, it returns a list of all data from the specified datasets.

    train_list's item format::

        {"image": image_file_path, "label": label_file_path, "dataset_name": dataset_name, "fold": fold_id}

    """
    if base_dirs is None:
        base_dirs = all_base_dirs  # all_base_dirs is the broader set
    # get the list of training/validation files (absolute path)
    json_files = get_json_files_k_folds(json_dir=json_dir, base_dirs=base_dirs)
    if datasets is None:
        loading_dict = json_files.copy()
    else:
        loading_dict = {k: look_up_option(k, json_files) for k in ensure_tuple(datasets)}
    train_list, val_list = [], []
    for k, j in loading_dict.items():
        t, v = datafold_read(j, basedir=all_base_dirs[k], fold=fold_idx, key=key)
        for item in t:
            item["dataset_name"] = k
        train_list += t
        for item in v:
            item["dataset_name"] = k
        val_list += v
    logger.warning(f"data list from datasets={datasets} fold={fold_idx}: train={len(train_list)}, val={len(val_list)}")
    return ensure_tuple(train_list), ensure_tuple(val_list)


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
        # mt.DataStatsd("label", additional_info=lambda x: x.meta["filename_or_obj"]),
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
        hist = hist[1 : (len(class_weights) + 1)]  # ignore background
        ws = np.array(class_weights)
        fg_weights = ws[hist > 0]
        # this volume has too many labels, treating it as less freq
        fg_w = fg_weights.max() if len(fg_weights) <= 10 else fg_weights.min()
        w.append(fg_w)
        item["w"] = fg_w
    return w


def quick_test(idx=5):
    """test timing for data loading, `python -m data.analyzer quick_test --idx=1700`"""
    idx = int(idx)
    b = time()
    xform = mt.Compose(get_data_stats_transforms())
    with WorkflowProfiler() as wp:
        for item in get_datalist_with_dataset_name()[0][idx - 3 : idx + 3]:
            a = time()
            output = xform(item)["label"]
            print(output.unique(), output.shape, output.meta["filename_or_obj"], time() - a)
    print("total time", time() - b)
    print(wp.get_times_summary_pd())


def calculate_dataset_weights(datalist):
    dataset_name = []
    dataset_counts = {}
    for item in datalist:
        dn = item['dataset_name']
        if dn in dataset_name:
            dataset_counts[dn] += 1
        else:
            dataset_name.append(dn)
            dataset_counts[dn] = 1
    dataset_weights = {}
    non_tumor_count = 0
    tumor_count = 0
    for item in dataset_name:
        if item not in ['Task03','Task06','Task07','Task08','Task10','Bone-NIH']:
            non_tumor_count += dataset_counts[item]
        else:
            tumor_count += dataset_counts[item]
            
    for item in dataset_name:
        if item not in ['Task03','Task06','Task07','Task08','Task10','Bone-NIH']:
            dataset_weights[item] = 100 / dataset_counts[item]# non_tumor_count
        else:
            dataset_weights[item] = 100 / dataset_counts[item] # tumor_count
            
    dataset_prob = {}
    total_prob = 0
    for item in dataset_name:
        dataset_prob[item] = dataset_weights[item] * dataset_counts[item]
        total_prob += dataset_prob[item]
    for item in dataset_name:
        dataset_prob[item] /= total_prob

    import json
    with open('./dataset_counts.yaml','w') as f:
        json.dump(dataset_counts, f, indent=4)        
    with open('./dataset_weights.yaml','w') as f:
        json.dump(dataset_weights, f, indent=4)        
    with open('./dataset_prob.yaml','w') as f:
        json.dump(dataset_prob, f, indent=4)   


if __name__ == "__main__":
    """
    run label histogram analyzer

        - `python -m data.analyzer compute_label_histogram_stats`
    """
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
