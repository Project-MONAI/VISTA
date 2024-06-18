# Description

Vista3D Foundation Model

## Dataset preparation
### Dataset format
All dataset needs a json file.

## Training

The training was performed with at least 32GB-memory GPUs. The training supports multi-node, multi-gpu training. VISTA3D has four stages training. The configurations represents the training procedure but cannot fully reproduce the weights of VISTA3D since each stage has multiple rounds with slightly varying configuration changes. 

### commands example

Execute model training:

```
export CUDA_VISIBLE_DEVICES=0; python -m scripts.train run --config_file "['configs/train/hyper_parameters_stage1.yaml']"
```

Execute multi-GPU model training (recommended):

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;torchrun --nnodes=1 --nproc_per_node=8 -m scripts.train run --config_file "['configs/train/hyper_parameters_stage1.yaml']" 
```
## Evaluation
We provide code for supported class fully automatic dice score evaluation (val_multigpu_point_patch), point click only (val_multigpu_point_patch), and auto + point (val_multigpu_autopoint_patch). Replace scripts.xxx with the corresponding script name and add in the dataset name.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;torchrun --nnodes=1 --nproc_per_node=8 -m scripts.val_multigpu_point_patch run --config_file "['configs/supported_eval/infer_patch_auto.yaml']" --dataset_name 'xxxx' 
```
For zero-shot, we perform iterative point sampling. 
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;torchrun --nnodes=1 --nproc_per_node=8 -m scripts.val_multigpu_point_iterative run --config_file "['configs/zero_shot/infer_iter_point_hcc.yaml']" 
```

## Inference
The inference is optimized in monai bundle. Use the bundle to perform inference or use NIM interface.  