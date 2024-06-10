#### Aggregating multiple datasets

This module by default assumes the datasets are mounted using these NGC batch commands:

```
--datasetid 1610565:/data/AbdomenCT-1K
--datasetid 1610636:/data/AMOS22
--datasetid 1610638:/data/BTCV
--datasetid 1610530:/data/CHAOS
--datasetid 1608298:/data/KiTS23
--datasetid 97292:/data/LiTS
--datasetid 83993:/data/Multi-organ-Abdominal-CT
--datasetid 1610587:/data/Pancreas-CT
--datasetid 1610781:/data/CT-ORG
--datasetid 68588:/data/Task06
--datasetid 68770:/data/Task07
--datasetid 68740:/data/Task08
--datasetid 69970:/data/Task09
--datasetid 68363:/data/Task10
--datasetid 1602467:/data/TotalSegmentator
--datasetid 110269:/data/WORD
```

Each dataset contains the original files that are publicly available and have been downloaded from external
and uploaded to NGC. Some datasets may have been minimally preprocessed by converting DICOM to NIfTI
(please see the details in each dataset folder).

The following steps are necessary for creating a multi-dataset data loader for model training.
Step 1 and 2 generate persistent JSON files, and only need to be run once when the JSON files don't exist.
Step 3 contains the Python modules that should be instantiated during each model training workflow.

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
