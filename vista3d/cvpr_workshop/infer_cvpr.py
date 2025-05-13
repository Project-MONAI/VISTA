import argparse
import glob

import monai
import monai.transforms
import nibabel as nib
import numpy as np
import torch
from monai.apps.vista3d.inferer import point_based_window_inferer
from monai.inferers import SlidingWindowInfererAdapt
from monai.networks.nets.vista3d import vista3d132
from monai.utils import optional_import

tqdm, _ = optional_import("tqdm", name="tqdm")
import os

from train_cvpr import ROI_SIZE

def convert_clicks(alldata):
    # indexes = list(alldata.keys())
    # data = [alldata[i] for i in indexes]
    data = alldata
    B = len(data)  # Number of objects
    indexes = np.arange(1, B + 1).tolist()
    # Determine the maximum number of points across all objects
    max_N = max(len(obj["fg"]) + len(obj["bg"]) for obj in data)

    # Initialize padded arrays
    point_coords = np.zeros((B, max_N, 3), dtype=int)
    point_labels = np.full((B, max_N), -1, dtype=int)

    for i, obj in enumerate(data):
        points = []
        labels = []

        # Add foreground points
        for fg_point in obj["fg"]:
            points.append(fg_point)
            labels.append(1)

        # Add background points
        for bg_point in obj["bg"]:
            points.append(bg_point)
            labels.append(0)

        # Fill in the arrays
        point_coords[i, : len(points)] = points
        point_labels[i, : len(labels)] = labels

    return point_coords, point_labels, indexes


if __name__ == "__main__":
    # set to true to save nifti files for visualization
    save_data = False
    point_inferer = True  # use point based inferen
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img_path", type=str, default="./tests")
    parser.add_argument("--save_path", type=str, default="./outputs/")
    parser.add_argument("--model", type=str, default="checkpoints/model_final.pth")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    # load model
    checkpoint_path = args.model
    model = vista3d132(in_channels=1)
    pretrained_ckpt = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(pretrained_ckpt, strict=True)

    # load data
    test_cases = glob.glob(os.path.join(args.test_img_path, "*.npz"))
    for img_path in test_cases:
        case_name = os.path.basename(img_path)
        print(case_name)
        img = np.load(img_path, allow_pickle=True)
        img_array = img["imgs"]
        spacing = img["spacing"]
        original_shape = img_array.shape
        affine = np.diag(spacing.tolist() + [1])  # 4x4 affine matrix
        if save_data:
            # Create a NIfTI image
            nifti_img = nib.Nifti1Image(img_array, affine)
            # Save the NIfTI file
            nib.save(nifti_img, img_path.replace(".npz", ".nii.gz"))
            nifti_img = nib.Nifti1Image(img["gts"], affine)
            # Save the NIfTI file
            nib.save(nifti_img, img_path.replace(".npz", "gts.nii.gz"))
        clicks = img.get("clicks", [{"fg": [[418, 138, 136]], "bg": []}])
        point_coords, point_labels, indexes = convert_clicks(clicks)
        # preprocess
        img_array = torch.from_numpy(img_array)
        img_array = img_array.unsqueeze(0)
        img_array = monai.transforms.ScaleIntensityRangePercentiles(
            lower=1, upper=99, b_min=0, b_max=1, clip=True
        )(img_array)
        img_array = img_array.unsqueeze(0)  # add channel dim
        device = "cuda"
        # slidingwindow
        with torch.no_grad():
            if not point_inferer:
                model.NINF_VALUE = 0  # set to 0 in case sliding window is used.
                # directly using slidingwindow inferer is not optimal.
                val_outputs = (
                    SlidingWindowInfererAdapt(
                        roi_size=ROI_SIZE,
                        sw_batch_size=1,
                        with_coord=True,
                        padding_mode="replicate",
                    )(
                        inputs=img_array.to(device),
                        transpose=True,
                        network=model.to(device),
                        point_coords=torch.from_numpy(point_coords).to(device),
                        point_labels=torch.from_numpy(point_labels).to(device),
                    )[
                        0
                    ]
                    > 0
                )
                final_outputs = torch.zeros_like(val_outputs[0], dtype=torch.float32)
                for i, v in enumerate(val_outputs):
                    final_outputs += indexes[i] * v
            else:
                # point based
                final_outputs = torch.zeros_like(img_array[0, 0], dtype=torch.float32)
                for i, v in enumerate(indexes):
                    val_outputs = (
                        point_based_window_inferer(
                            inputs=img_array.to(device),
                            roi_size=ROI_SIZE,
                            transpose=True,
                            with_coord=True,
                            predictor=model.to(device),
                            mode="gaussian",
                            sw_device=device,
                            device=device,
                            center_only=True,  # only crop the center
                            point_coords=torch.from_numpy(point_coords[[i]]).to(device),
                            point_labels=torch.from_numpy(point_labels[[i]]).to(device),
                        )[0]
                        > 0
                    )
                    final_outputs[val_outputs[0]] = v
                final_outputs = torch.nan_to_num(final_outputs)
            # save data
            if save_data:
                # Create a NIfTI image
                nifti_img = nib.Nifti1Image(
                    final_outputs.to(torch.float32).data.cpu().numpy(), affine
                )
                # Save the NIfTI file
                nib.save(
                    nifti_img,
                    os.path.join(args.save_path, case_name.replace(".npz", ".nii.gz")),
                )
            np.savez_compressed(
                os.path.join(args.save_path, case_name),
                segs=final_outputs.to(torch.float32).data.cpu().numpy(),
            )
