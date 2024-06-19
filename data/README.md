#### Aggregating multiple datasets

The training workflow requires one or multiple dataset JSON files to specifiy the image and segmentation pairs as well as dataset preprocessing transformations.
Example files are located in the `data/jsons` folder.

The JSON file has the following structure:
```python
{
    "training": [
        {
            "image": "img1.nii.gz",  # relative path to the primary image file
            "label": "label1.nii.gz",  # optional relative path to the primary label file
            "pseudo_label": "p_label1.nii.gz",  # optional relative path to the pseudo label file
            "pseudo_label_reliability": 1  # optional reliability score for pseudo label
            "label_sv": "label_sv1.nii.gz",  # optional relative path to the supervoxel label file
            "fold": 0  # optional fold index for cross validation, fold 0 is used for training
        },

        ...
    ],
    "training_transform": [
        # a set of monai transform configuration for dataset-specific loading
    ]
    "original_label_dict": {"1": "liver", ...},
    "label_dict": {"1": "liver", ...}
}
```

During training, the JSON files will be consumed along with additional configurations, for example:
```py
from data.datasets import get_datalist_with_dataset_name_and_transform

train_files, _, dataset_specific_transforms, dataset_specific_transforms_val = \
    get_datalist_with_dataset_name_and_transform(
        datasets=train_datasets,
        fold_idx=fold,
        image_key=image_key,
        label_key=label_key,
        label_sv_key=label_sv_key,
        pseudo_label_key=pseudo_label_key,
        num_patches_per_image=parser.get_parsed_content("num_patches_per_image"),
        patch_size=parser.get_parsed_content("patch_size"),
        json_dir=json_dir)
```

The following steps are necessary for creating a multi-dataset data loader for model training.
Step 1 and 2 generate persistent JSON files based on the original dataset (the `image` and `label` pairs; without the additional pseudo label or supervoxel-based label), and only need to be run once when the JSON files don't exist.
Step 3 is optional for generating overall data analysis stats.

##### 1. Generate data list JSON file
```
python -m data.make_datalists
```

This script reads image and label folders, lists all the nii.gz files,
creates a JSON file in a format:

```json
{
    "training": [
        {"image": "img0001.nii.gz", "label": "label0001.nii.gz", "fold": 0},
        {"image": "img0002.nii.gz", "label": "label0002.nii.gz", "fold": 2},
        ...
    ],
    "testing": [
        {"image": "img0003.nii.gz", "label": "label0003.nii.gz"},
        {"image": "img0004.nii.gz", "label": "label0004.nii.gz"},
        ...
    ]
    "original_label_dict": {"1": "liver", ...},
    "label_dict": {"1": "liver", ...}
}
```

This step includes a 5-fold cross validation splitting and
some logic for 80-20 training/testing splitting.

The `original_label_dict` corresponds to the original dataset label definitions.
The `label_dict` modifies `original_label_dict` by simply rephrasing the terms.
For example in Task06, `cancer` is renamed to `lung tumor`.
The output of this step is multiple JSON files, each file corresponds
to one dataset.


##### 2. Verify data pairs and generate a global label dictionary
```
python -m data.datasets
```

This script computes a super set of labels from all the dataset JSON files.
The output of this step is a `jsons/label_dict.json` file,
representing the global label dictionary mapping, from class names to globally unique class indices (integers).


##### 3. Compute class frequencies, data transform utilities
```
python -m data.analyzer ...
```

This file (`data/analyzer.py`) contains useful transforms for reading images
and labels, converting labels from dataset-specific labels to the global labels
according to `jsons/label_dict.json`.
