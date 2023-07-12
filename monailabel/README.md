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

# MONAI VISTA Models 

<div align="center"> <img src="./assets/imgs/wholeBody.png" width="800"/> </div>


### Table of Contents
- [Overview](#Overview)
- [Start MONAI VISTA with MONAI Label](#MONAI-Label-Integration)
  - [Step 1. Installation](#Installation)
  - [Step 2. MONAI Label monaivista app](#MONAI-VISTA-APP)
  - [Step 3. MONAI VISTA - Label Plugins](#MONAI-VISTA-Viewer-Plugins)
  - [Step 4. Data Preparation](#Sample-Data)
  - [Step 5. Start MONAI Label Server and Start Annotating!](#Step-5-Start-MONAI-Label-Server-and-Start-Annotating)
- [Tutorials](#MONAI-Label-Tutorials)
- [Contributing](#Contributing)
- [Community](#Community)

## Overview

MONAI VISTA is a plaform solution of deploying medical segmentation foundation models. This section
 provides a MONAI Label integration of APIs and samples apps. The integration is a server-client 
 system that facilitates interactive medical image segmentation using AI models such as segment 
 anything medical model (SAMM) or other prompt-based annotation algorithms. 

MONAI VISTA - MONAI Label integration is an intelligent opem source ecosystem that embables users
to create and deploy vision foundation models especially for medical segmenation. It provides 
interfaces of class- and point-prompts that AI models can take as input. The integration also provides
sample 3D Slicer plugin UIs.

## MONAI Label Integration

### Installation

MONAI VISTA models are integrated based on [MONAI Label](https://docs.monai.io/projects/label/en/latest/index.html#).
Start using MONAI Label locally and run installlation with your familiar visualization tools. 
Stable version software represents the currently tested and supported visualization tools with 
latest release of MONAI Label. Weekly preview version is available if users want the latest feature, 
not fully tested.

Refer to [MONAI Label installation](https://docs.monai.io/projects/label/en/latest/installation.html) page
for details. 

For milestone release, users can install from PyPl with command:

```bash
pip install monailabel

```

For Docker and Github installation, refer to MONAI Label [Github](https://github.com/Project-MONAI/MONAILabel)

### MONAI VISTA APP

Based on MONAI Label, MONAI VISTA is developed as an app. This app has example models 
to do both interactive and "Everything" segmentation over medical images. 
Prompt-based segment experience is highlighted. Including class prompts and point click prompts, 
Segmentation with latest deep learning architectures (e.g., Segmentation Anything Model (SAM)) for multiple lung, abdominal, and pelvis
organs. Interactive tools includes comptrol points, class prompt check boxes are developed with viewer plugins. 

The MONAI VISTA app contains several tasks:

- Inferencing tasks: These tasks allow end-users to invoke pre-trained models for image analysis.
- Training and fine-tuning tasks: TBD

#### Step 1. Implementing an Inference Task
To implement an inference task, developers must inherit the  [InferTask](https://github.com/Project-MONAI/MONAILabel/blob/main/monailabel/tasks/infer/basic_infer.py) interface, which specifies a list of pre- and post-transforms and an inferer.

The code snippet below demonstrates an example implementation of `InferTask`. In this example, the image is pre-processed to a Numpy array and input into the `SimpleInferer`. The resulting output is post-processed by applying sigmoid activation with binary discretization.

<pre style="background: #f4f4f4; border: 1px solid #ddd; border-left: 3px solid #02a3a3; line-height: 1.6; padding: 1.5em;">
from monai.inferers import SimpleInferer
from monai.transforms import (LoadImaged, ToNumpyd, Activationsd AsDiscreted, ToNumpyd)
from monailabel.interfaces.tasks import InferTask

class MyInfer(InferTask):
  def pre_transforms(self, data=None):
      return [
          LoadImaged(keys="image"),
          ToNumpyd(keys="image"),
      ]
  def inferer(self, data=None):
      return SimpleInferer()

  def post_transforms(self, data=None):
      return [
          Activationsd(keys="pred", sigmoid=True),
          AsDiscreted(keys="pred", threshold=0.5),
          ToNumpyd(keys="pred"),
      ]
</pre>

Note that `inferer` needs to be defined by developers.

For more details of `monaivista` app, see the [sample-app page](https://github.com/Project-MONAI/VISTA/tree/add_monailabel_integration/monailabel/monaivista)

### MONAI VISTA Viewer Plugins


<div align="center"> <img src="./assets/imgs/3dslicer_plugin.png" width="500"/> </div>



### Sample Data


### Start MONAI Label Server with VISTA Model

