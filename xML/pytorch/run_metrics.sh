## 완전 초기 val 한 것
# python calc_metrics_new.py \
#   --dataset_path ../data/ \
#   --dataset transfer \
#   --split val \
#   --model_path ./finetuning_output/new/transfer/2025-07-29_14-21-53/2025-07-29_14-21-53_finetuned.pth \
#   --exp_dir   ./finetuning_output/new/transfer/2025-07-29_14-21-53/explanations \
#   --output_csv ./metrics_내맘/metrics_transfer.csv \
#   --classifier resnet18 \
#   --init_bias 3.0 \
#   --model_class ExplainerClassifierCNN \
#   --img_size 224 224



## 수정해서 test로 한 것
python calc_metrics_new.py \
  --dataset_path ../data/ \
  --dataset rehistogan \
  --split test \
  --model_path ./finetuning_output/new/rehistogan/resnet18/2025-08-30_16-46-26_phase2_best_loss.pt \
  --exp_dir   ./test_outputs/rehistogan_resnet18/explanations \
  --output_csv ./metrics_내맘/metrics_rehistogan_resnet18.csv \
  --classifier resnet18 \
  --init_bias 3.0 \
  --model_class ExplainerClassifierCNN \
  --img_size 224 224

