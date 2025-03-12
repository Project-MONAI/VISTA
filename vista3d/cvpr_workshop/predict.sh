#!/bin/bash

# Run inference script with input/output folder paths
python3 infer.py --test_img_path /workspace/inputs/ --save_path /workspace/outputs/ --model /workspace/model_final.pth
