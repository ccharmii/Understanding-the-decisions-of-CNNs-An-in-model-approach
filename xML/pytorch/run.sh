#!/bin/bash

## run.sh

# ## train - 논문 그대로는 아니지만 추측 + GPT
# python train.py \
# --dataset imagenetHVZ \
# --dataset_path '../datasets/imagenet_data' \
# --nr_epochs 10,10,53 -bs 8 \
# --init_bias 3.0 \
# --loss hybrid \
# --alpha 1.0,0.25,0.9 \
# --beta 0.9 \
# --gamma 1.0 -lr_clf 0.01,0.01,0.01 -lr_expl 0.01,0.01,0.01 \
# --aug_prob 0.2 \
# --opt sgd -clf resnet18 \
# --early_patience 100,100,10 \
# --folder './pth_논문'


## train - 깃허브 그대로
python train.py \
--dataset imagenetHVZ \
--dataset_path '../data/imagenet_data' \
--nr_epochs 10,10,60 -bs 8 \
--init_bias 3.0 \
--loss hybrid \
--alpha 1.0,0.25,0.9 \
--beta 0.9 \
--gamma 1.0 -lr_clf 0.01,0,0.01 -lr_expl 0,0.01,0.01 \
--aug_prob 0.2 \
--opt sgd -clf resnet101 \
--early_patience 100,100,10 \
--folder './pth'


## test
# python test.py ./pth/2025-06-01_05-37-17/2025-06-01_05-37-17_phase0_latest.pt \
# --dataset imagenetHVZ \
# --dataset_path '../datasets/imagenet_data' \
# --alpha 0.9 -bs 8 -clf resnet18 \
# --init_bias 3.0 \
# --loss hybrid \
# --beta 0.9 \
# --gamma 1.0