#!/bin/bash
python main_2pt5d.py --max_epochs 100 --val_every 1 --optim_lr 0.000005 \
--num_patch 24 --num_prompt 32 \
--json_list ./totalsegmentator_104organs_folds_v2.json \
--data_dir /data/ \
--roi_z_iter 9 --save_checkpoint \
--sam_base_model vit_b \
--logdir finetune_ckpt_example --point_prompt --label_prompt --distributed --seed 12346 \
--iterative_training_warm_up_epoch 50 --reuse_img_embedding \
--label_prompt_warm_up_epoch 25 \
--checkpoint ./runs/9s_2dembed_model.pt \
--num_classes 105
