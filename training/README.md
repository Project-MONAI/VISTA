# Model Overview
![image](../assets/img.png)
This repository contains the training code for MONAI VISTA 2.5D model. MONAI VISTA 2.5D is based on SAM [1] but we finetune
the model (image encoder, prompt encoder, and mask decoder) on 3D medical data. MONAI VISTA  introduces
the class-label prompt and enables the fully automatic inference on known classes. It also shows the potential of
generalizing to unknown class. In addition, MONAI VISTA takes 2.5D input, so our model can leverage the information
from multiple slices.


# Works in progress
We are still actively developing this model. Features coming soon:
1. **MONAI VISTA 3D Model**. It will support 3D volumetric inputs to enable a larger field of view and reduce user’s annotation efforts.
2. **Text-based class-label prompt**. It will support encoding input text (e.g., “A computerized tomography of {Liver}”) as the class-label prompt.
3. **Multiple Datasets Training**. We are working on supporting more pre-defined class labels for the fully automatic inference pipeline. Due to the nature of prompt-based segmentation, our model is compatible with the partial label training.


# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

# Models

Please download the pre-trained weights from this
<a href="https://drive.google.com/file/d/1ozJMe8hkLJfhNEJz-IHvV_tpyW3T2r_E/view?usp=sharing"> link</a>.

# Data Preparation
![image](../assets/img_1.png)
Figure source from the TotalSegmentator [2].

The training data is from the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) [2].

- Target: 104 anatomical structures.
- Task: Segmentation
- Modality: CT
- Size: 1204 3D volumes
- Spacing: [1.5, 1.5, 1.5]

More details about preprocessing this dataset can be found at
<a href="https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation#preprocessing"> link</a>.

The json file containing the data list that is used to train our models can be downloaded from
<a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/totalsegmentator_104organs_folds_v2.json"> link</a>.


Note that you need to provide the location of your dataset directory and json file by using ```--data_dir``` and ```--json_list```.

# Training

A MONAI VISTA 2.5D model (ViT-B base) with standard hyperparameters is defined as:

```py
_build_vista2pt5d(
        encoder_in_chans=27,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,
        image_size=1024,
        clip_class_label_prompt=False,
        patch_embed_3d=False,
    )
```

Or, you may directly call:

```py
build_vista2pt5d_vit_b()
```

The above VISTA 2.5D model is used for CT images (9 slices 2.5D) with input spacing size ```(1.5, 1.5, 1.5)``` and for ```104``` class promptable segmentation.

Using the default values for hyperparameters,
the following command can be used to initiate training using PyTorch native AMP package:

``` bash
python main_2pt5d.py --max_epochs 100 --val_every 1 --optim_lr 0.000005 \
--num_patch 24 --num_prompt 32 \
--json_list ./totalsegmentator_104organs_folds_v2.json \
--data_dir /data/ \
--roi_z_iter 9 --save_checkpoint \
--sam_base_model vit_b \
--logdir finetune_ckpt_example --point_prompt --label_prompt --distributed --seed 12346 \
--iterative_training_warm_up_epoch 50 --reuse_img_embedding \
--label_prompt_warm_up_epoch 25 \
--checkpoint ./runs/9s_2dembed_model.pt
```

Above command will start the finetune training for the provided pre-trained weights
(50 epochs single-step training and 50 epochs iterative training).

# Evaluation

To evaluate the `VISTA 2.5D model` using MONAI Label, please find the detailed instructions from
<a href="https://github.com/Project-MONAI/VISTA"> here</a>.


# Reference

```
[1]: @article{kirillov2023segany,
      title={Segment Anything},
      author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
      journal={arXiv:2304.02643},
      year={2023}
    }

[2]: @article{wasserthal2022totalsegmentator,
      title={TotalSegmentator: robust segmentation of 104 anatomical structures in CT images},
      author={Wasserthal, Jakob and Meyer, Manfred and Breit, Hanns-Christian and Cyriac, Joshy and Yang, Shan and Segeroth, Martin},
      journal={arXiv preprint arXiv:2208.05868},
      year={2022}
    }
```
