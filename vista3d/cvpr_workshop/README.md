<!--
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Overview
This repository is written for the "CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation"([link](https://www.codabench.org/competitions/5263/)) challenge. It
is based on MONAI 1.4. Many of the functions in the main VISTA3D repository are moved to MONAI 1.4 and this simplified folder will directly use components from MONAI.

It is simplified to train interactive segmentation models across different modalities. The sophisticated transforms and recipes used for VISTA3D are removed. The finetuned VISTA3D checkpoint on the challenge subsets is available [here](https://drive.google.com/file/d/1r2KvHP_30nHR3LU7NJEdscVnlZ2hTtcd/view?usp=sharing)

# Setup
```
pip install -r requirements.txt
```

# Training
Download the challenge subsets finetuned [checkpoint](https://drive.google.com/file/d/1hQ8imaf4nNSg_43dYbPSJT0dr7JgAKWX/view?usp=sharing) or VISTA3D original [checkpoint]((https://drive.google.com/file/d/1DRYA2-AI-UJ23W1VbjqHsnHENGi0ShUl/view?usp=sharing)). Generate a json list that contains your traning data and update the json file path in the script.
```
torchrun --nnodes=1 --nproc_per_node=8 train_cvpr.py
```
The checkpoint saved by train_cvpr.py can be updated by `update_ckpt.py` to remove the additional `module` key due to multi-gpu training.


# Inference
You can directly download the [docker file](https://drive.google.com/file/d/1r2KvHP_30nHR3LU7NJEdscVnlZ2hTtcd/view?usp=sharing) for the challenge baseline.
We provide a Dockerfile to satisfy the challenge format. For more details, refer to the [challenge website]((https://www.codabench.org/competitions/5263/)).
```
docker build -t vista3d:latest .
docker save -o vista3d.tar.gz vista3d:latest
```
You can also directly run `predict.sh`. Download the finetuned checkpoint and modify the `--model=/your_downloaded_checkpoint`. Change `save_data=True` in `infer_cvpr.py` to save predictions to nifti files for visualization.
