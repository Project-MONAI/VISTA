# python3 -m scripts.export --config_file 'configs/infer.yaml' - infer_everything --image_file 'example-1.nii.gz'

python3 -m scripts.export --config_file 'configs/infer.yaml' - infer --image_file 'example-1.nii.gz' --label_prompt [1] --save_mask true
