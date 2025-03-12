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

It is overly simplied to train interactive segmentation models across different modalities. The sophisticated transforms and recipes used for VISTA3D are removed. 

# Setup
```
pip install -r requirements.txt
```

# Training
Download VISTA3D pretrained checkpoint or from scratch. Generate a json list that contains your traning data.
```
torchrun --nnodes=1 --nproc_per_node=8 train_cvpr.py
```

# Inference
We provide a Dockerfile to satisfy the challenge format. For more details, refer to the [challenge website]((https://www.codabench.org/competitions/5263/))


