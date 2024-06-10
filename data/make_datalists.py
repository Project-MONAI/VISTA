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


# Step 1.
# reading image and label folders, listing all the nii.gz files,
# creating a data_list.json for training and validation
# the data_list.json format is like ('testing' labels are optional):
# {
#     "training": [
#         {"image": "img0001.nii.gz", "label": "label0001.nii.gz", "fold": 0},
#         {"image": "img0002.nii.gz", "label": "label0002.nii.gz", "fold": 2},
#         ...
#     ],
#     "testing": [
#         {"image": "img0003.nii.gz", "label": "label0003.nii.gz"},
#         {"image": "img0004.nii.gz", "label": "label0004.nii.gz"},
#         ...
#     ]
# }

import os
import re
from pprint import pprint

from monai.apps import get_logger
from monai.bundle import ConfigParser
from monai.data.utils import partition_dataset

logger = get_logger(__name__)
test_ratio = 0.2  # test split
n_folds = 5  # training and validation split
seed = 20230808  # random seed for data partition_dataset reproducibility
output_json_dir = os.path.join(os.path.dirname(__file__), "jsons")


def register_make(func):
    """
    register the function to make the data list
    """
    global _make_funcs
    if "_make_funcs" not in globals():
        _make_funcs = {}
    _make_funcs[func.__name__] = func
    return func


def search_image_files(base_dir, ext, regex=None):
    """returns a list of relative filenames with given extension in `base_dir`"""
    print(f"searching ext={ext} from base_dir={base_dir}")
    images = []
    for root, _, files in os.walk(base_dir):
        images.extend(os.path.join(root, filename) for filename in files if filename.endswith(ext))
    if regex is not None:
        images = [x for x in images if re.compile(regex).search(x) is not None]
    print(f"found {len(images)} *.{ext} files")
    return sorted(images)


def create_splits_and_write_json(
    images, labels, ratio, num_folds, json_name, rng_seed, label_dict, original_label_dict=None
):
    """
    first generate training/test split, then from the training part,
    generate training/validation num_folds
    """
    items = [{"image": img, "label": lab} for img, lab in zip(images, labels)]
    train_test = partition_dataset(items, ratios=[1 - ratio, ratio], shuffle=True, seed=rng_seed)
    print(f"training: {len(train_test[0])}, testing: {len(train_test[1])}")
    train_val = partition_dataset(train_test[0], num_partitions=num_folds, shuffle=True, seed=rng_seed)
    print(f"training validation folds sizes: {[len(x) for x in train_val]}")
    training = []
    for f, x in enumerate(train_val):
        for item in x:
            item["fold"] = f
            training.append(item)

    # write json
    parser = ConfigParser({})
    parser["training"] = training
    parser["testing"] = train_test[1]

    parser["label_dict"] = label_dict
    parser["original_label_dict"] = original_label_dict or label_dict

    print(f"writing {json_name}\n\n")
    if os.path.exists(json_name):
        raise ValueError(f"please remove existing datalist file: {json_name}")
    ConfigParser.export_config_file(parser.config, json_name, indent=4)


def filtering_files(base_url, image_names, label_names, idx=-1):
    """
    check the idx-th item in the lists of image and label filenames, remove:

        - image files without corresponding label files

    """
    _tmp_img = os.path.join(base_url, image_names[idx])
    _tmp_lab = os.path.join(base_url, label_names[idx])
    if not (os.path.exists(_tmp_img) and os.path.exists(_tmp_lab)):
        if not os.path.exists(_tmp_img):
            logger.warning(f"image file {_tmp_img} pair does not exist")
        if not os.path.exists(_tmp_lab):
            logger.warning(f"label file {_tmp_lab} pair does not exist")
        image_names.pop(idx)
        label_names.pop(idx)


####
@register_make
def make_abdomenct_1k():
    base_url = "/data/AbdomenCT-1K"
    dataset_name = "AbdomenCT-1K"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "Mask"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"Case_(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"Case_{idx}_0000.nii.gz"
        for f in ["AbdomenCT-1K-ImagePart1", "AbdomenCT-1K-ImagePart2", "AbdomenCT-1K-ImagePart3"]:
            if os.path.exists(os.path.join(base_url, f, img_name)):
                images.append(os.path.join(f, img_name))
                break
        # print(f"image: {images[-1]}, label: {labels[-1]}")
        filtering_files(base_url, images, labels)
    label_dict = {1: "liver", 2: "kidney", 3: "spleen", 4: "pancreas"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_flare22():
    base_url = "/data/AbdomenCT-1K/FLARE22Train"
    dataset_name = "FLARE22"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "labels"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"FLARE22_Tr_(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"FLARE22_Tr_{idx}_0000.nii.gz"
        images.append(os.path.join("images", img_name))
        filtering_files(base_url, images, labels)
    label_dict = {
        1: "liver",
        2: "right kidney",
        3: "spleen",
        4: "pancreas",
        5: "aorta",
        6: "inferior vena cava",
        7: "right adrenal gland",
        8: "left adrenal gland",
        9: "gallbladder",
        10: "esophagus",
        11: "stomach",
        12: "duodenum",
        13: "left kidney",
    }
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_amos22():
    base_url = "/data/AMOS22"
    dataset_name = "AMOS22"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "labelsTr"), ".nii.gz")
    masks += search_image_files(os.path.join(base_url, "labelsVa"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"amos_(\d+).nii.gz").search(rel_mask)[1]
        if int(idx) >= 500:  # skip the MRI cases
            labels.pop()
            continue
        img_name = f"amos_{idx}.nii.gz"
        for f in ["imagesTr", "imagesVa"]:
            if os.path.exists(os.path.join(base_url, f, img_name)):
                images.append(os.path.join(f, img_name))
                break
        # print(f"image: {images[-1]}, label: {labels[-1]}")
        filtering_files(base_url, images, labels)
    label_dict = {
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "aorta",
        9: "postcava",
        10: "pancreas",
        11: "right adrenal gland",
        12: "left adrenal gland",
        13: "duodenum",
        14: "bladder",
        15: "prostate or uterus",
    }
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_btcv_abdomen():
    base_url = "/data/BTCV/Abdomen"
    dataset_name = "BTCV-Abdomen"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "RawData", "Training", "label"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"label(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"img{idx}.nii.gz"
        images.append(os.path.join("RawData", "Training", "img", img_name))
        filtering_files(base_url, images, labels)
    label_dict = {
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "aorta",
        9: "inferior vena cava",
        10: "portal vein and splenic vein",
        11: "pancreas",
        12: "right adrenal gland",
        13: "left adrenal gland",
    }
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_btcv_cervix():
    base_url = "/data/BTCV/Cervix"
    dataset_name = "BTCV-Cervix"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "RawData", "Training", "label"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"(\d+)-Mask.nii.gz").search(rel_mask)[1]
        img_name = f"{idx}-Image.nii.gz"
        images.append(os.path.join("RawData", "Training", "img", img_name))
        filtering_files(base_url, images, labels)
    label_dict = {1: "bladder", 2: "uterus", 3: "rectum", 4: "small bowel"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_chaos():
    base_url = "/data/CHAOS"
    dataset_name = "CHAOS"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "Train_Sets_nifti_ct"), "segmentation.nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"(\d+)_segmentation.nii.gz").search(rel_mask)[1]
        img_name = f"{idx}_image.nii.gz"
        images.append(os.path.join("Train_Sets_nifti_ct", img_name))
        filtering_files(base_url, images, labels)
    label_dict = {1: "liver"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_ct_org():
    base_url = "/data/CT-ORG"
    dataset_name = "CT-ORG"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "OrganSegmentations"), ".nii.gz", regex=r"labels")
    images, labels = [], []
    for mask in masks:
        idx = re.compile(r"labels-(\d+).nii.gz").search(mask)[1]
        if idx in {"19", "70", "74", "76"}:
            continue  # these are problematic cases
        _fixed_mask = os.path.join(base_url, "fixed_affine", f"labels-{idx}.nii.gz")
        img_name = f"volume-{idx}.nii.gz"
        mask_name = f"labels-{idx}.nii.gz"
        if os.path.exists(_fixed_mask):  # there are newer fixed files
            images.append(os.path.join("fixed_affine", img_name))
            labels.append(os.path.join("fixed_affine", mask_name))
        else:
            images.append(os.path.join("OrganSegmentations", img_name))
            labels.append(os.path.join("OrganSegmentations", mask_name))
        filtering_files(base_url, images, labels)
    original_label_dict = {
        1: "liver",
        2: "bladder",
        3: "lungs",
        4: "kidneys",
        5: "bone",
        6: "brain",
    }
    label_dict = {
        1: "liver",
        2: "bladder",
        3: "lung",
        4: "kidney",
        5: "bone",
        6: "brain",
    }
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict, original_label_dict)


####
@register_make
def make_kits23():
    base_url = "/data/KiTS23/dataset"
    dataset_name = "KiTS23"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(base_url, "segmentation.nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"case_(\d+)").search(rel_mask)[1]
        img_name = f"case_{idx}"
        images.append(os.path.join(img_name, "imaging.nii.gz"))
        filtering_files(base_url, images, labels)
    original_label_dict = {1: "kidney", 2: "tumor", 3: "cyst"}
    label_dict = {1: "kidney", 2: "kidney tumor", 3: "kidney cyst"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict, original_label_dict)


####
@register_make
def make_lits23():
    base_url = "/data/LiTS/Training_Batch"
    dataset_name = "LiTS"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(base_url, ".nii", regex=r"segmentation")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"segmentation-(\d+).nii").search(rel_mask)[1]
        img_name = f"volume-{idx}.nii"
        images.append(img_name)
        filtering_files(base_url, images, labels)
    original_label_dict = {1: "liver", 2: "tumor"}
    label_dict = {1: "liver", 2: "liver tumor"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict, original_label_dict)


####
@register_make
def make_multi_organ_btcv():
    base_url = "/data/Multi-organ-Abdominal-CT/res_1.0mm_relabeled2"
    dataset_name = "Multi-organ-Abdominal-CT-btcv"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "label_btcv_multiorgan"), ".nii")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"label(\d+).nii").search(rel_mask)[1]
        img_name = f"img{idx}.nii"
        images.append(os.path.join("images_btcv", img_name))
        filtering_files(base_url, images, labels)
    label_dict = {
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "aorta",
        9: "inferior vena cava",
        10: "portal vein and splenic vein",
        11: "pancreas",
        12: "right adrenal gland",
        13: "left adrenal gland",
        14: "duodenum",
    }
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_multi_organ_tcia():
    base_url = "/data/Multi-organ-Abdominal-CT/res_1.0mm_relabeled2"
    dataset_name = "Multi-organ-Abdominal-CT-tcia"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "label_tcia_multiorgan+rkidney"), ".nii")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"label(\d+).nii").search(rel_mask)[1]
        img_name = f"PANCREAS_{idx}.nii"
        images.append(os.path.join("images_tcia", img_name))
        filtering_files(base_url, images, labels)
    label_dict = {
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "pancreas",
        9: "duodenum",
    }
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_pancreas_ct():
    base_url = "/data/Pancreas-CT"
    dataset_name = "Pancreas-CT"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "TCIA_pancreas_labels-02-05-2017"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"label(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"PANCREAS_{idx}.nii.gz"
        images.append(os.path.join("manifest-1599750808610", "nifti", img_name))
        filtering_files(base_url, images, labels)
    label_dict = {1: "pancreas"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_task06():
    base_url = "/data/Task06"
    dataset_name = "Task06"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "labelsTr"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"lung_(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"lung_{idx}.nii.gz"
        images.append(os.path.join("imagesTr", img_name))
        filtering_files(base_url, images, labels)
    original_label_dict = {1: "cancer"}
    label_dict = {1: "lung tumor"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict, original_label_dict)


####
@register_make
def make_task07():
    base_url = "/data/Task07"
    dataset_name = "Task07"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "labelsTr"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"pancreas_(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"pancreas_{idx}.nii.gz"
        images.append(os.path.join("imagesTr", img_name))
        filtering_files(base_url, images, labels)
    original_label_dict = {1: "pancreas", 2: "cancer"}
    label_dict = {1: "pancreas", 2: "pancreatic tumor"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict, original_label_dict)


####
@register_make
def make_task08():
    base_url = "/data/Task08"
    dataset_name = "Task08"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "labelsTr"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"hepaticvessel_(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"hepaticvessel_{idx}.nii.gz"
        images.append(os.path.join("imagesTr", img_name))
        filtering_files(base_url, images, labels)
    original_label_dict = {1: "Vessel", 2: "Tumour"}
    label_dict = {1: "hepatic vessel", 2: "hepatic tumor"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict, original_label_dict)


####
@register_make
def make_task09():
    base_url = "/data/Task09"
    dataset_name = "Task09"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "labelsTr"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"spleen_(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"spleen_{idx}.nii.gz"
        images.append(os.path.join("imagesTr", img_name))
        filtering_files(base_url, images, labels)
    label_dict = {1: "spleen"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_task10():
    base_url = "/data/Task10"
    dataset_name = "Task10"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "labelsTr"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"colon_(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"colon_{idx}.nii.gz"
        images.append(os.path.join("imagesTr", img_name))
        filtering_files(base_url, images, labels)
    label_dict = {1: "colon cancer primaries"}
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict)


####
@register_make
def make_total_segmentator():
    base_url = "/data/TotalSegmentator"
    dataset_name = "TotalSegmentator"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "labels"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"s(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"s{idx}.nii.gz"
        images.append(os.path.join("images", img_name))
        filtering_files(base_url, images, labels)
    original_label_dict = (
        {  # https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            4: "gallbladder",
            5: "liver",
            6: "stomach",
            7: "aorta",
            8: "inferior_vena_cava",
            9: "portal_vein_and_splenic_vein",
            10: "pancreas",
            11: "adrenal_gland_right",
            12: "adrenal_gland_left",
            13: "lung_upper_lobe_left",
            14: "lung_lower_lobe_left",
            15: "lung_upper_lobe_right",
            16: "lung_middle_lobe_right",
            17: "lung_lower_lobe_right",
            18: "vertebrae_L5",
            19: "vertebrae_L4",
            20: "vertebrae_L3",
            21: "vertebrae_L2",
            22: "vertebrae_L1",
            23: "vertebrae_T12",
            24: "vertebrae_T11",
            25: "vertebrae_T10",
            26: "vertebrae_T9",
            27: "vertebrae_T8",
            28: "vertebrae_T7",
            29: "vertebrae_T6",
            30: "vertebrae_T5",
            31: "vertebrae_T4",
            32: "vertebrae_T3",
            33: "vertebrae_T2",
            34: "vertebrae_T1",
            35: "vertebrae_C7",
            36: "vertebrae_C6",
            37: "vertebrae_C5",
            38: "vertebrae_C4",
            39: "vertebrae_C3",
            40: "vertebrae_C2",
            41: "vertebrae_C1",
            42: "esophagus",
            43: "trachea",
            44: "heart_myocardium",
            45: "heart_atrium_left",
            46: "heart_ventricle_left",
            47: "heart_atrium_right",
            48: "heart_ventricle_right",
            49: "pulmonary_artery",
            50: "brain",
            51: "iliac_artery_left",
            52: "iliac_artery_right",
            53: "iliac_vena_left",
            54: "iliac_vena_right",
            55: "small_bowel",
            56: "duodenum",
            57: "colon",
            58: "rib_left_1",
            59: "rib_left_2",
            60: "rib_left_3",
            61: "rib_left_4",
            62: "rib_left_5",
            63: "rib_left_6",
            64: "rib_left_7",
            65: "rib_left_8",
            66: "rib_left_9",
            67: "rib_left_10",
            68: "rib_left_11",
            69: "rib_left_12",
            70: "rib_right_1",
            71: "rib_right_2",
            72: "rib_right_3",
            73: "rib_right_4",
            74: "rib_right_5",
            75: "rib_right_6",
            76: "rib_right_7",
            77: "rib_right_8",
            78: "rib_right_9",
            79: "rib_right_10",
            80: "rib_right_11",
            81: "rib_right_12",
            82: "humerus_left",
            83: "humerus_right",
            84: "scapula_left",
            85: "scapula_right",
            86: "clavicula_left",
            87: "clavicula_right",
            88: "femur_left",
            89: "femur_right",
            90: "hip_left",
            91: "hip_right",
            92: "sacrum",
            93: "face",
            94: "gluteus_maximus_left",
            95: "gluteus_maximus_right",
            96: "gluteus_medius_left",
            97: "gluteus_medius_right",
            98: "gluteus_minimus_left",
            99: "gluteus_minimus_right",
            100: "autochthon_left",
            101: "autochthon_right",
            102: "iliopsoas_left",
            103: "iliopsoas_right",
            104: "urinary_bladder",
        }
    )
    label_dict = {
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "aorta",
        8: "inferior vena cava",
        9: "portal vein and splenic vein",
        10: "pancreas",
        11: "right adrenal gland",
        12: "left adrenal gland",
        13: "left lung upper lobe",
        14: "left lung lower lobe",
        15: "right lung upper lobe",
        16: "right lung middle lobe",
        17: "right lung lower lobe",
        18: "vertebrae L5",
        19: "vertebrae L4",
        20: "vertebrae L3",
        21: "vertebrae L2",
        22: "vertebrae L1",
        23: "vertebrae T12",
        24: "vertebrae T11",
        25: "vertebrae T10",
        26: "vertebrae T9",
        27: "vertebrae T8",
        28: "vertebrae T7",
        29: "vertebrae T6",
        30: "vertebrae T5",
        31: "vertebrae T4",
        32: "vertebrae T3",
        33: "vertebrae T2",
        34: "vertebrae T1",
        35: "vertebrae C7",
        36: "vertebrae C6",
        37: "vertebrae C5",
        38: "vertebrae C4",
        39: "vertebrae C3",
        40: "vertebrae C2",
        41: "vertebrae C1",
        42: "esophagus",
        43: "trachea",
        44: "heart myocardium",
        45: "left heart atrium",
        46: "left heart ventricle",
        47: "right heart atrium",
        48: "right heart ventricle",
        49: "pulmonary artery",
        50: "brain",
        51: "left iliac artery",
        52: "right iliac artery",
        53: "left iliac vena",
        54: "right iliac vena",
        55: "small bowel",
        56: "duodenum",
        57: "colon",
        58: "left rib 1",
        59: "left rib 2",
        60: "left rib 3",
        61: "left rib 4",
        62: "left rib 5",
        63: "left rib 6",
        64: "left rib 7",
        65: "left rib 8",
        66: "left rib 9",
        67: "left rib 10",
        68: "left rib 11",
        69: "left rib 12",
        70: "right rib 1",
        71: "right rib 2",
        72: "right rib 3",
        73: "right rib 4",
        74: "right rib 5",
        75: "right rib 6",
        76: "right rib 7",
        77: "right rib 8",
        78: "right rib 9",
        79: "right rib 10",
        80: "right rib 11",
        81: "right rib 12",
        82: "left humerus",
        83: "right humerus",
        84: "left scapula",
        85: "right scapula",
        86: "left clavicula",
        87: "right clavicula",
        88: "left femur",
        89: "right femur",
        90: "left hip",
        91: "right hip",
        92: "sacrum",
        93: "face",
        94: "left gluteus maximus",
        95: "right gluteus maximus",
        96: "left gluteus medius",
        97: "right gluteus medius",
        98: "left gluteus minimus",
        99: "right gluteus minimus",
        100: "left autochthon",
        101: "right autochthon",
        102: "left iliopsoas",
        103: "right iliopsoas",
        104: "bladder",
    }
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict, original_label_dict)


####
@register_make
def make_word():
    base_url = "/data/WORD"
    dataset_name = "WORD"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "labelsTr"), ".nii.gz")
    masks += search_image_files(os.path.join(base_url, "labelsTs"), ".nii.gz")
    masks += search_image_files(os.path.join(base_url, "labelsVal"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"word_(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"word_{idx}.nii.gz"
        for f in ["imagesTr", "imagesTs", "imagesVal"]:
            if os.path.exists(os.path.join(base_url, f, img_name)):
                images.append(os.path.join(f, img_name))
                break
        filtering_files(base_url, images, labels)
    original_label_dict = {
        1: "liver",
        2: "spleen",
        3: "left_kidney",
        4: "right_kidney",
        5: "stomach",
        6: "gallbladder",
        7: "esophagus",
        8: "pancreas",
        9: "duodenum",
        10: "colon",
        11: "intestine",
        12: "adrenal",
        13: "rectum",
        14: "bladder",
        15: "Head_of_femur_L",
        16: "Head_of_femur_R",
    }
    label_dict = {
        1: "liver",
        2: "spleen",
        3: "left kidney",
        4: "right kidney",
        5: "stomach",
        6: "gallbladder",
        7: "esophagus",
        8: "pancreas",
        9: "duodenum",
        10: "colon",
        11: "intestine",
        12: "adrenal gland",
        13: "rectum",
        14: "bladder",
        15: "left head of femur",
        16: "right head of femur",
    }
    create_splits_and_write_json(images, labels, test_ratio, n_folds, json_name, seed, label_dict, original_label_dict)


if __name__ == "__main__":
    pprint(_make_funcs)
    for func_name, f in _make_funcs.items():
        print(f"running {func_name}")
        f()
