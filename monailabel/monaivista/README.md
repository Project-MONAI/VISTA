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

# TODO add readme

The MONAI VISTA app contains several tasks:

- Inferencing tasks: These tasks allow end-users to invoke pre-trained models for image analysis.
- Training and fine-tuning tasks: TBD

#### Implementing an Inference Task
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
