import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import inspect  # ★ 추가

import utils
import losses
from Dataset import Dataset, load_data
from ExplainerClassifierCNN import ExplainerClassifierCNN
from EarlyStopping import EarlyStopping
from summary import summary

torch.manual_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description="finetuning 스크립트")
    # 데이터 및 모델 I/O
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="blastocyst",
                        choices=["synthetic","NIH-NCI","imagenetHVZ","hmc","kromp","wavect","embryo","rehistogan","rehistogan_ft","stainnet","stainnet_ft"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--folder", type=str, default="./finetune_out")
    parser.add_argument("--gpu", type=str, default="1")

    # 네트워크 설정
    parser.add_argument("-clf", "--classifier", type=str, default="resnet50",
                        choices=["vgg","resnet18","resnet34","resnet50","resnet101","resnet152"])
    parser.add_argument("--init_bias", type=float, default=2.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_explainer", action="store_true")
    parser.add_argument("--freeze_classifier", action="store_true")

    # 손실 & 학습 파라미터
    parser.add_argument("--loss", type=str, default="unsupervised", choices=["unsupervised","hybrid"])
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--alpha", type=str, default="1.0,0.0,0.9")

    parser.add_argument("--nr_epochs", type=str, default="10,10,50")
    parser.add_argument("--lr_clf", type=str, default="1e-3,0,1e-4")
    parser.add_argument("--lr_expl", type=str, default="0,1e-4,1e-4")

    parser.add_argument("--opt", type=str, default="sgd", choices=["sgd","adadelta"])
    parser.add_argument("--decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--factor", type=float, default=0.2)

    parser.add_argument("-bs","--batch_size", type=int, default=32)
    parser.add_argument("--img_size", nargs=2, type=int, default=[224,224])
    parser.add_argument("--aug_prob", type=float, default=0)

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early_patience", type=str, default="200,200,200")
    parser.add_argument("--early_delta", type=float, default=1e-4)
    return parser.parse_args()


# ★ PyTorch 2.6 호환 안전 로더
def safe_torch_load(path, device):
    # 신뢰된 체크포인트 전제: numpy 객체 허용
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype])
    except Exception:
        pass

    # weights_only 지원 시 True→실패하면 False 폴백
    if "weights_only" in inspect.signature(torch.load).parameters:
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception as e:
            print("[WARN] weights_only=True 로드 실패 → False 재시도:", e)
            return torch.load(path, map_location=device, weights_only=False)
    else:
        return torch.load(path, map_location=device)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    timestamp, path = utils.create_folder(args.folder)
    with open(os.path.join(path, timestamp + "_config.txt"), "w") as f:
        f.write(str(vars(args)))

    masks = (args.loss == "hybrid")
    if args.dataset == "embryo" and masks:
        print("Warning: embryo has no masks, switching to unsupervised.")
        masks = False

    tr_df, val_df, test_df, weights, classes = load_data(
        folder=args.dataset_path, dataset=args.dataset, masks=masks, class_weights="balanced"
    )
    if test_df is None:
        test_df = val_df

    weights = torch.FloatTensor(weights).to(device)

    train_loader = DataLoader(
        Dataset(tr_df, preprocess=utils.norm, masks=masks, img_size=tuple(args.img_size), aug_prob=args.aug_prob),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        Dataset(val_df, preprocess=utils.norm, masks=masks, img_size=tuple(args.img_size), aug_prob=0),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = ExplainerClassifierCNN(
        num_classes=len(classes), img_size=tuple(args.img_size),
        clf=args.classifier, init_bias=args.init_bias, pretrained=args.pretrained
    ).to(device)

    # ★ 체크포인트 로드에 안전 로더 사용
    if args.checkpoint:
        ck = safe_torch_load(args.checkpoint, device)
        if isinstance(ck, dict) and "classifier" in ck and "explainer" in ck:
            model.classifier.load_state_dict(ck["classifier"])
            model.explainer.load_state_dict(ck["explainer"])
        else:
            model.load_state_dict(ck)
        print(f"Restored checkpoint: {args.checkpoint}")

    if args.freeze_explainer:
        utils.freeze(model.explainer)
    if args.freeze_classifier:
        utils.freeze(model.classifier)

    summary(model.classifier, [(3,)+tuple(args.img_size),(1,)+tuple(args.img_size)],
            filename=os.path.join(path, timestamp+"_classifier.txt"))
    summary(model.explainer, (3,)+tuple(args.img_size),
            filename=os.path.join(path, timestamp+"_explainer.txt"))

    nr_epochs = [int(x) for x in args.nr_epochs.split(",")]
    lr_clf   = [float(x) for x in args.lr_clf.split(",")]
    lr_expl  = [float(x) for x in args.lr_expl.split(",")]
    alpha    = [float(x) for x in args.alpha.split(",")]
    early_pat= [int(x) for x in args.early_patience.split(",")]

    for phase in range(3):
        if nr_epochs[phase] == 0:
            continue
        print(f"\n=== PHASE {phase} ===")
        early_stopping = EarlyStopping(
            patience=early_pat[phase], delta=args.early_delta,
            verbose=True, folder=path, timestamp=timestamp+f"_phase{phase}"
        )

        params = [
            {"params": model.classifier.parameters(), "lr": lr_clf[phase]},
            {"params": model.explainer.parameters(), "lr": lr_expl[phase]}
        ]
        if args.opt=="adadelta":
            opt = optim.Adadelta(params, weight_decay=args.decay)
        else:
            opt = optim.SGD(params, weight_decay=args.decay, momentum=args.momentum)

        # (일부 버전에서 verbose 인자 에러) → verbose 제거
        if "resnet" in args.classifier.lower():
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=args.factor, patience=args.patience,
                min_lr=args.min_learning_rate,
            )
        else:
            scheduler = None

        epochs_run = 0
        history_file = os.path.join(path, timestamp+f"_{phase}_history.csv")
        with open(history_file, "w") as f:
            f.write("epoch,train_global,train_cls,train_exp,train_acc,val_global,val_cls,val_exp,val_acc\n")

        for epoch in range(nr_epochs[phase]):
            epochs_run += 1
            print(f"Phase{phase} Epoch {epoch+1}/{nr_epochs[phase]}")
            model.train(train_loader, opt, device, args, phase, weights, alpha[phase])
            train_global_loss, train_explainer_loss, train_classifier_loss, train_classifier_acc = \
                model.validation(train_loader, device, args, alpha[phase])
            val_global_loss, val_explainer_loss, val_classifier_loss, val_classifier_acc = \
                model.validation(val_loader, device, args, alpha[phase])
            print(f" Train loss={train_global_loss:.4f} cls={train_classifier_loss:.4f} exp={train_explainer_loss:.4f} acc={train_classifier_acc:.4f}")
            print(f" Val   loss={val_global_loss:.4f} cls={val_classifier_loss:.4f} exp={val_explainer_loss:.4f} acc={val_classifier_acc:.4f}\n")

            model.checkpoint(os.path.join(path, timestamp+f"_phase{phase}"),
                             epoch, val_global_loss, val_classifier_acc, opt)

            with open(history_file, "a") as f:
                f.write(f"{epoch},{train_global_loss:.4f},{train_classifier_loss:.4f},{train_explainer_loss:.4f},{train_classifier_acc:.4f},{val_global_loss:.4f},{val_classifier_loss:.4f},{val_explainer_loss:.4f},{val_classifier_acc:.4f}\n")

            early_stopping(val_global_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_global_loss)

        utils.plot_metric_train_val(epochs_run, history_file, "cls", path, f"phase{phase}_cls_loss.png", "Classifier Loss")

        # ★ phase별 best ckpt 로드도 안전 로더 사용
        ckpt_path = os.path.join(path, f"{timestamp}_phase{phase}_best_loss.pt")
        ckpt = safe_torch_load(ckpt_path, device)
        model.classifier.load_state_dict(ckpt["classifier"])
        model.explainer.load_state_dict(ckpt["explainer"])
        model.save_explanations(val_loader, phase, device, path,
                                classes=[str(c) for c in classes],
                                cmap="viridis", only_explanations=True)

    final = os.path.join(path, f"{timestamp}_finetuned.pth")
    torch.save({"classifier":model.classifier.state_dict(),
                "explainer":model.explainer.state_dict()}, final)
    print(f"Finetuning 완료: {final}")


if __name__=="__main__":
    main()
