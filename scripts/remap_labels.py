import argparse
import os
from typing import Dict

import SimpleITK as sitk

MAP = {
    0: 0,  # background
    1: 105,  # brainstem
    2: 106,  # opticChiasm
    3: 107,  # opticNerveL
    4: 108,  # opticNerveR
    5: 109,  # parotidGlandL
    6: 110,  # garotidGlandR
    7: 111,  # mandible
}


def remap_labels(data_dir: str, file_substr_id: str, mapping: Dict[int, int], reverse: bool = False):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file_substr_id in file:
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                label_itk = sitk.ReadImage(file_path, sitk.sitkUInt32)
                label_array = sitk.GetArrayFromImage(label_itk)
                if reverse:
                    for k, v in mapping.items():
                        label_array[label_array == v] = k
                else:
                    for k, v in mapping.items():
                        label_array[label_array == k] = v
                new_label_itk = sitk.GetImageFromArray(label_array)
                new_label_itk.CopyInformation(label_itk)
                sitk.WriteImage(new_label_itk, file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="data directory")
    parser.add_argument("--file_substr_id", default="segmentation", help="substr used to indentify the label file")
    args = parser.parse_args()
    remap_labels(args.data_dir, args.file_substr_id, MAP, reverse=False)


if __name__ == "__main__":
    main()
