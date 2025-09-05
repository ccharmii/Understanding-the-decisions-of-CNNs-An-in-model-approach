#!/usr/bin/env bash
# run_finetuning.sh: finetuning.py 실행 예제 스크립트


python finetuning.py \
  --dataset embryo \
  --dataset_path "../data/" \
  --checkpoint "./finetuning_pth/resnet34_best_loss.pt" \
  --folder "./finetuning_output/new/embryo" \
  --init_bias 3.0 \
  --loss unsupervised \
  --alpha 1.0,0.25,0.9 \
  --beta 0.9 \
  --gamma 0.5 \
  --lr_clf 0.01,0,0.01 \
  --lr_expl 0,0.0001,0.0001 \
  --aug_prob 0.2 \
  --opt sgd -clf resnet34 \
  --batch_size 8 \
  --early_patience 100,100,10 \

## 7/28 phase 1에서 explainer가 천천히 움직이도록 lr_expl 낮춤
## --> 원래: lr_expl 0,0.01,0.01
