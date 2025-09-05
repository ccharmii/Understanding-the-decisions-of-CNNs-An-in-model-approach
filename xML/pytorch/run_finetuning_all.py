import os

# data_lst = ["embryo", "hmc", "kromp", "wavect", "rehistogan", "rehistogan_ft", "stainnet", "stainnet_ft"]
# data_lst = ["stainnet", "stainnet_ft"]

# for d in data_lst:
#     print('\n\n-- ', d, '--------------------------------------------')
#     os.system(
#         f"python finetuning.py "
#         f"--dataset {d} "
#         f"--dataset_path ../data/ "
#         f"--checkpoint ./finetuning_pth/resnet18_best_loss.pt "
#         f"--folder ./finetuning_output/new/{d} "
#         f"--init_bias 3.0 "
#         f"--loss unsupervised "
#         f"--alpha 1.0,0.25,0.9 "
#         f"--beta 0.9 "
#         f"--gamma 0.5 "
#         f"--lr_clf 0.01,0,0.01 "
#         f"--lr_expl 0,0.0001,0.0001 "
#         f"--aug_prob 0.2 "
#         f"--opt sgd "
#         f"-clf resnet18 "
#         f"--batch_size 8 "
#         f"--early_patience 100,100,10"
#     )


data_lst = ["embryo", "wavect", "rehistogan", "rehistogan_ft", "stainnet", "stainnet_ft"]

for d in data_lst:
    print('\n\n-- ', d, '--------------------------------------------')
    os.system(
        f"python finetuning.py "
        f"--dataset {d} "
        f"--dataset_path ../data/ "
        f"--checkpoint ./finetuning_pth/resnet34_best_loss.pt "
        f"--folder ./finetuning_output/new/{d} "
        f"--init_bias 3.0 "
        f"--loss unsupervised "
        f"--alpha 1.0,0.25,0.9 "
        f"--beta 0.9 "
        f"--gamma 0.5 "
        f"--lr_clf 0.01,0,0.01 "
        f"--lr_expl 0,0.0001,0.0001 "
        f"--aug_prob 0.2 "
        f"--opt sgd "
        f"-clf resnet34 "
        f"--batch_size 8 "
        f"--early_patience 100,100,10"
    )