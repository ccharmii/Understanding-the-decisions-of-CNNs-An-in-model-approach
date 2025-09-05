import os


# data_lst = ["embryo", "hmc", "kromp", "wavect", "rehistogan", "rehistogan_ft", "stainnet", "stainnet_ft"]

# data_lst = ["wavect", "rehistogan", "rehistogan_ft", "stainnet", "stainnet_ft"]

# for d in data_lst:
#     print('\n\n-- ', d, '--------------------------------------------')
#     os.system(
#         f"python test.py "
#         f"--model_ckpt ./finetuning_output/pth/resnet18/{d}.pt "
#         f"--dataset {d} "
#         f"--dataset_path ../data/ "
#         f"-bs 8 "
#         f"-clf resnet18 "
#         f"--init_bias 3.0 "
#         f"--loss unsupervised "
#         f"--alpha 0.9 "
#         f"--beta 0.9 "
#         f"--gamma 1.0 "
#         f"--out_dir ./test_outputs/{d}_resnet18 "
#     )

data_lst = ["embryo", "wavect", "rehistogan", "rehistogan_ft", "stainnet", "stainnet_ft"]


for d in data_lst:
    print('\n\n-- ', d, '--------------------------------------------')
    os.system(
        f"python test.py "
        f"--model_ckpt ./finetuning_output/pth/resnet34/{d}.pt "
        f"--dataset {d} "
        f"--dataset_path ../data/ "
        f"-bs 8 "
        f"-clf resnet34 "
        f"--init_bias 3.0 "
        f"--loss unsupervised "
        f"--alpha 0.9 "
        f"--beta 0.9 "
        f"--gamma 1.0 "
        f"--out_dir ./test_outputs/{d}_resnet34 "
    )
