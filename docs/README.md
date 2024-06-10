# Description

Vista3D Foundation Model

## Dataset preparation
### Dataset format
All dataset is 

## Training

The training was performed with at least 32GB-memory GPUs. The training supports multi-node, multi-gpu training. VISTA3D has four stages training. The configurations may not fully

## commands example

Execute model training:

```
export CUDA_VISIBLE_DEVICES=0; python -m scripts.train run --config_file "['configs/train/hyper_parameters_point_all.yaml']"
```

Execute multi-GPU model training (recommended):

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;torchrun --nnodes=1 --nproc_per_node=8 -m scripts.train run --config_file "['configs/train/hyper_parameters_point_all.yaml']" 
```

Execute validation:
Change the drop_label, drop_point value in hyperparameters.yaml to change validation scheme.
```
export CUDA_VISIBLE_DEVICES=0; python -m scripts.val_multigpu run --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_train.yaml','configs/transforms_infer.yaml','configs/transforms_validate.yaml']" 
```
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; torchrun --nnodes=1 --nproc_per_node=8 -m scripts.val_multigpu run --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_train.yaml','configs/transforms_infer.yaml','configs/transforms_validate.yaml']" 
```
To do iterative validation where false postive/negative points are sampled from automatic segmentation,
```
export CUDA_VISIBLE_DEVICES=0; python -m scripts.val_multigpu_iterative run --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_train.yaml','configs/transforms_infer.yaml','configs/transforms_validate.yaml']" 
```
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; torchrun --nnodes=1 --nproc_per_node=8 -m scripts.val_multigpu_iterative run --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_train.yaml','configs/transforms_infer.yaml','configs/transforms_validate.yaml']" 
```
Execute inference:
Segment class from 0-104, just set output_classes
```
export CUDA_VISIBLE_DEVICES=0; python -m scripts.infer run_everything_auto \
 --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_train.yaml','configs/transforms_infer.yaml','configs/transforms_validate.yaml']" \
 --image_file "/your_path_to_file/file_name.nii.gz" --output_classes 105 
```
Segment specific classes, set class_prompt. The class_prompt need to match the represented class index (e.g. 1 means spleen)
```
export CUDA_VISIBLE_DEVICES=0; python -m scripts.infer run_everything_auto \
 --config_file "['configs/hyper_parameters.yaml','configs/network.yaml','configs/transforms_train.yaml','configs/transforms_infer.yaml','configs/transforms_validate.yaml']" \
 --image_file "/your_path_to_file/file_name.nii.gz" --class_prompt [1,2,3,4]
```
Execute interactive:
```
python -m scripts.interactive run
```
