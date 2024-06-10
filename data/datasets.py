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

import argparse
import os
import warnings
from pprint import pformat

from monai.apps import get_logger
from monai.bundle import ConfigParser

logger = get_logger(__name__)
cur_json_dir = os.path.join(os.path.dirname(__file__), "jsons")

# Step 2: compute dataset json files
# each dataset contains a folder of images and a folder of labels
# this is for internal testing only, not for public release because of the data license
all_base_dirs = {
    "AbdomenCT-1K": "/data/AbdomenCT-1K",  # Y
    "FLARE22": "/data/AbdomenCT-1K/FLARE22Train",  # Y
    "AMOS22": "/data/AMOS22",  # Y
    "BTCV-Abdomen": "/data/BTCV/Abdomen",  # Y
    "BTCV-Cervix": "/data/BTCV/Cervix",  # Y
    "CHAOS": "/data/CHAOS",  # X incompatible license
    "CT-ORG": "/data/CT-ORG",  # Y
    "KiTS23": "/data/KiTS23/dataset",  # X incompatible license
    "LiTS": "/data/LiTS/Training_Batch",  # X incompatible license
    "Multi-organ-Abdominal-CT-btcv": "/data/Multi-organ-Abdominal-CT/res_1.0mm_relabeled2",  # Y (BTCV sub)
    "Multi-organ-Abdominal-CT-tcia": "/data/Multi-organ-Abdominal-CT/res_1.0mm_relabeled2",  # Y (Pancreas-CT sub)
    "Pancreas-CT": "/data/Pancreas-CT",  # Y
    "Task06": "/data/Task06",  # Y
    "Task07": "/data/Task07",  # Y
    "Task08": "/data/Task08",  # Y
    "Task09": "/data/Task09",  # Y
    "Task10": "/data/Task10",  # Y
    "TotalSegmentator": "/data/TotalSegmentator",  # Y
    "WORD": "/data/WORD",  # X incompatible license
    "Task03": "/data/Task03",
    "Bone-NIH": "/data/Bone-NIH",
    "TotalSegmentatorV2": "/data/TotalSegmentatorV2",
    "C4KC-KiTS": "/data/C4KC-KiTS/nifti",  # continual learning story
    "CRLM-CT": "/data/CRLM-CT/nifti",
    "VerSe": "/data/VerSe/",
    "AeroPath": "/data/AeroPath/",
    "Autopet23": "/data/Autopet23/",  # Y
    "LIDC-IDRI": "/data/LIDC-IDRI/",  # Y
    "CTPelvic1K-CLINIC": "/data/CTPelvic1K-CLINIC",  # Y
    "COLON_ACRIN6664": "/data/COLON_ACRIN6664",  # Y
    "NLST": "/data/NLST",
    "LIDC": "/data/LIDC",
    "Covid19": "/data/Covid19",
    "TCIA_Colon": "/data/TCIA_Colon",
    "StonyBrook-CT": "/data/StonyBrook-CT"
}

valid_base_dirs_v1 = {  # the keys are used to form the json file name. totalseg v1
    "AbdomenCT-1K": "/data/AbdomenCT-1K",
    "FLARE22": "/data/AbdomenCT-1K/FLARE22Train",
    "AMOS22": "/data/AMOS22",
    "BTCV-Abdomen": "/data/BTCV/Abdomen",
    "BTCV-Cervix": "/data/BTCV/Cervix",
    "CT-ORG": "/data/CT-ORG",
    "Multi-organ-Abdominal-CT-btcv": "/data/Multi-organ-Abdominal-CT/res_1.0mm_relabeled2",
    "Multi-organ-Abdominal-CT-tcia": "/data/Multi-organ-Abdominal-CT/res_1.0mm_relabeled2",
    "Pancreas-CT": "/data/Pancreas-CT",
    "Task06": "/data/Task06",
    "Task07": "/data/Task07",
    "Task08": "/data/Task08",
    "Task09": "/data/Task09",
    "Task10": "/data/Task10",
    "TotalSegmentator": "/data/TotalSegmentator",
    "Task03": "/data/Task03",
    "Bone-NIH": "/data/Bone-NIH",
    "C4KC-KiTS": "/data/C4KC-KiTS/nifti",  # continual learning story
}

valid_base_dirs = {  # the keys are used to form the json file name.
    "AbdomenCT-1K": "/data/AbdomenCT-1K",
    "FLARE22": "/data/AbdomenCT-1K/FLARE22Train",
    "AMOS22": "/data/AMOS22",
    "BTCV-Abdomen": "/data/BTCV/Abdomen",
    "BTCV-Cervix": "/data/BTCV/Cervix",
    "CT-ORG": "/data/CT-ORG",
    "Multi-organ-Abdominal-CT-btcv": "/data/Multi-organ-Abdominal-CT/res_1.0mm_relabeled2",
    "Multi-organ-Abdominal-CT-tcia": "/data/Multi-organ-Abdominal-CT/res_1.0mm_relabeled2",
    "Pancreas-CT": "/data/Pancreas-CT",
    "Task06": "/data/Task06",
    "Task07": "/data/Task07",
    "Task08": "/data/Task08",
    "Task09": "/data/Task09",
    "Task10": "/data/Task10",
    "TotalSegmentator": "/data/TotalSegmentator",
    "TotalSegmentatorV2": "/data/TotalSegmentatorV2",
    "Task03": "/data/Task03",
    "Bone-NIH": "/data/Bone-NIH",
    "C4KC-KiTS": "/data/C4KC-KiTS/nifti",  # continual learning story
    "CRLM-CT": "/data/CRLM-CT/nifti",
    "VerSe": "/data/VerSe/",
    "AeroPath": "/data/AeroPath/",
    "Autopet23": "/data/Autopet23/",
    "LIDC-IDRI": "/data/LIDC-IDRI/",
    "CTPelvic1K-CLINIC": "/data/CTPelvic1K-CLINIC",
    "COLON_ACRIN6664": "/data/COLON_ACRIN6664",
    "NLST": "/data/NLST",
    "LIDC": "/data/LIDC",
    "Covid19": "/data/Covid19",
    "TCIA_Colon": "/data/TCIA_Colon",
    "StonyBrook-CT": "/data/StonyBrook-CT"
}

pseudo_base_dirs = {  # the keys are used to form the json file name.
    "AbdomenCT-1K": "/data/AbdomenCT-1K",
    "FLARE22": "/data/AbdomenCT-1K/FLARE22Train",
    "AMOS22": "/data/AMOS22",
    "BTCV-Abdomen": "/data/BTCV/Abdomen",
    "BTCV-Cervix": "/data/BTCV/Cervix",
    "CT-ORG": "/data/CT-ORG",
    "Multi-organ-Abdominal-CT-btcv": "/data/Multi-organ-Abdominal-CT/res_1.0mm_relabeled2",
    "Multi-organ-Abdominal-CT-tcia": "/data/Multi-organ-Abdominal-CT/res_1.0mm_relabeled2",
    "Pancreas-CT": "/data/Pancreas-CT",
    "Task06": "/data/Task06",
    "Task07": "/data/Task07",
    "Task08": "/data/Task08",
    "Task09": "/data/Task09",
    "Task10": "/data/Task10",
    "TotalSegmentator": "/data/TotalSegmentator",
    "TotalSegmentatorV2": "/data/TotalSegmentatorV2",
    "Task03": "/data/Task03",
    "Bone-NIH": "/data/Bone-NIH",
    "CRLM-CT": "/data/CRLM-CT/nifti",
    "VerSe": "/data/VerSe/",
    "AeroPath": "/data/AeroPath/",
    "NLST": "/data/NLST",
    "LIDC": "/data/LIDC",
    "Covid19": "/data/Covid19",
    "TCIA_Colon": "/data/TCIA_Colon",
    "StonyBrook-CT": "/data/StonyBrook-CT"
}



def get_json_files_k_folds(json_dir=None, base_dirs=None, k=5):
    """the json files are generated by data/make_datalists.py, stored at `json_dir`"""
    if json_dir is None:
        json_dir = cur_json_dir
    if base_dirs is None:
        base_dirs = valid_base_dirs
    output_dict = {item: os.path.join(json_dir, f"{item}_{k}_folds.json") for item in base_dirs}
    logger.debug(pformat(output_dict))
    return output_dict


def get_label_dict_k_folds(json_dir=None, base_dirs=None, k=5):
    """
    the input json files are generated by data/make_datalists.py
    the output is a dictionary of global label mapping, from class name to class index.
    """
    if json_dir is None:
        json_dir = cur_json_dir
    json_output = os.path.join(json_dir, "label_dict.json")
    if os.path.exists(json_output):
        logger.warning(f"reading local label dict from {json_output}")
        parser = ConfigParser.load_config_file(json_output)
        return dict(parser)
    jsons = get_json_files_k_folds(json_dir=json_dir, base_dirs=base_dirs, k=k)
    # load all label directories
    label_dict = []
    for _, j in jsons.items():
        if not os.path.exists(j):
            raise RuntimeError(f"cannot find dataset json file {j}")
        parser = ConfigParser.load_config_file(j)
        label_dict += list(parser["label_dict"].items())
    # remove duplicated label class names
    labels = []
    for _, name in label_dict:
        if name not in labels:
            labels.append(name)
        else:
            warnings.warn(f"duplicated label name {name}", stacklevel=2)
    output_dict = {name: idx + 1 for idx, name in enumerate(labels)}
    logger.debug(f"writing label dict to {json_output}")
    ConfigParser.export_config_file(output_dict, json_output, indent=4)
    parser = ConfigParser.load_config_file(json_output)
    return dict(parser)


def get_class_names(json_dir=None, base_dirs=None):
    """
    the list of class names, background is at 0
    """
    parser = ConfigParser.load_config_file(os.path.join(json_dir, "label_dict.json"))
    label_dict = dict(parser)
    # label_dict = get_label_dict_k_folds(json_dir=json_dir, base_dirs=base_dirs)
    label_dict["unspecified region"] = 0
    inv_label_dict = {v: k for k, v in label_dict.items()}
    label_list = []
    for i in range(len(label_dict)):
        label_list.append(inv_label_dict[i])
    return label_list


if __name__ == "__main__":
    """
    directly run this script to verify that:
    0. all json files exist
    1. all images and labels exist
    2. no duplicated images, labels
    """
    # read command line arguments
    cli_args = argparse.ArgumentParser(description="check dataset json files")
    cli_args.add_argument("--base_dirs", "-b", default=None, help="base directories of datasets")
    cli_args = cli_args.parse_args()
    base_dirs = valid_base_dirs if cli_args.base_dirs is None else all_base_dirs
    # go through all json files to check if all images and labels exist and no duplicated item
    all_images, all_labels = [], []
    for k, j in get_json_files_k_folds(json_dir=None, base_dirs=base_dirs).items():
        b = base_dirs[k]
        if not os.path.exists(j):
            raise RuntimeError(f"cannot find {j}")
        parser = ConfigParser.load_config_file(j)
        for x in parser["training"] + parser["testing"]:
            image = os.path.join(b, x["image"])
            label = os.path.join(b, x["label"])
            if not os.path.exists(image):
                raise RuntimeError(f"cannot find {image}")
            if not os.path.exists(label):
                raise RuntimeError(f"cannot find {label}")
            print(image, label)

    # check no duplicated in all_images
    if len(all_images) != len(set(all_images)):
        raise RuntimeError("duplicated image files")
    if len(all_labels) != len(set(all_labels)):
        raise RuntimeError("duplicated label files")
    get_label_dict_k_folds(json_dir=None, base_dirs=base_dirs, k=5)