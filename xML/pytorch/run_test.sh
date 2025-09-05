#!/usr/bin/env bash


## test

python test.py \
    --model_ckpt ./finetuning_output/new/rehistogan/resnet18/2025-08-30_16-46-26_phase2_best_loss.pt \
    --dataset_path '../data/' \
    --dataset rehistogan \
    --alpha 0.9 -bs 8 -clf resnet18 \
    --init_bias 3.0 \
    --loss unsupervised \
    --beta 0.9 \
    --gamma 1.0 \
    --out_dir './test_outputs/rehistogan'

