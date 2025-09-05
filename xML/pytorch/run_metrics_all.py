import os

# data_lst = ["embryo", "hmc", "kromp", "wavect", "rehistogan", "rehistogan_ft", "stainnet", "stainnet_ft"]
# data_lst = ["wavect", "rehistogan", "rehistogan_ft", "stainnet", "stainnet_ft"]
# for d in data_lst:
#     print('\n\n-- ', d, '--------------------------------------------')
#     os.system(
#         f"python calc_metrics_new.py "
#         f"--dataset {d} "
#         f"--dataset_path ../data/ "
#         f"--split test "
#         f"--model_path ./finetuning_output/pth/resnet18/{d}.pt "
#         f"--exp_dir ./test_outputs/{d}_resnet18/explanations "
#         f"--output_csv ./metrics_내맘/metrics_{d}_resnet18.csv "
#         f"--classifier resnet18 "
#         f"--init_bias 3.0 "
#         f"--model_class ExplainerClassifierCNN "
#         f"--img_size 224 224 "
#         )

data_lst = ["embryo", "wavect", "rehistogan", "rehistogan_ft", "stainnet", "stainnet_ft"]
for d in data_lst:
    print('\n\n-- ', d, '--------------------------------------------')
    os.system(
        f"python calc_metrics_new.py "
        f"--dataset {d} "
        f"--dataset_path ../data/ "
        f"--split test "
        f"--model_path ./finetuning_output/pth/resnet34/{d}.pt "
        f"--exp_dir ./test_outputs/{d}_resnet34/explanations "
        f"--output_csv ./metrics_내맘/metrics_{d}_resnet34.csv "
        f"--classifier resnet34 "
        f"--init_bias 3.0 "
        f"--model_class ExplainerClassifierCNN "
        f"--img_size 224 224 "
        )