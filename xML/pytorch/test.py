## test.py



import os
import sys
import argparse
import inspect
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils                           # norm, plot_* 등 유틸
from ExplainerClassifierCNN import ExplainerClassifierCNN
from Dataset import Dataset, load_data

# 재현성
torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Explainer+Classifier 테스트 (SS 예측)")

    # 실행 환경
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")

    # 체크포인트 / 출력 경로
    parser.add_argument("--model_ckpt", type=str, required=True,
                        help="모델 체크포인트(.pth/.pt) 경로")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="결과 저장 폴더(기본: <ckpt_dir>/<ckpt_dirname>)")

    # 데이터셋
    parser.add_argument("--dataset_path", type=str, required=True, help="데이터셋 루트 폴더")
    parser.add_argument("--dataset", type=str, default="imagenetHVZ",
                        choices=["synthetic","NIH-NCI","imagenetHVZ","hmc","kromp","wavect","embryo","rehistogan","rehistogan_ft","stainnet","stainnet_ft"],
                        help="load_data()에 넘길 이름")
    parser.add_argument("--nr_classes", type=int, default=2, help="타겟 클래스 수")
    parser.add_argument("--img_size", nargs=2, type=int, default=[224, 224], help="H W")

    # 배치/백본/초기 bias
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-clf", "--classifier", type=str, default="resnet50",
                        choices=["vgg", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    parser.add_argument("--init_bias", type=float, default=1.0)

    # 손실 파라미터
    parser.add_argument("--loss", type=str, default="unsupervised", choices=["unsupervised", "hybrid"])
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float)

    # 히트맵 컬러맵
    parser.add_argument("--cmap", type=str, default="viridis")

    return parser.parse_args()


# --- PyTorch 2.6 호환: 안전 로더(allowlist + weights_only 우선) ---
def load_checkpoint_safely(path, device):
    # 신뢰된 ckpt라는 전제 하에 필요한 타입 allowlist
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype])
    except Exception:
        pass

    can_use_wo = "weights_only" in inspect.signature(torch.load).parameters
    if can_use_wo:
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception as e:
            print("[WARN] weights_only=True 로드 실패 → False 재시도:", e)
            return torch.load(path, map_location=device, weights_only=False)
    else:
        return torch.load(path, map_location=device)


def main():
    args = parse_args()

    # 마스크 없는 데이터셋은 hybrid 금지 → unsupervised로 강제
    if args.dataset in ("blastocyst", "embryo") and args.loss == "hybrid":
        print(f"Warning: '{args.dataset}' has no masks; switching to unsupervised loss.")
        args.loss = "unsupervised"

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 경로 설정
    ckpt_path = args.model_ckpt
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_dirname = os.path.basename(ckpt_dir)

    # 지정 안 하면 예전과 동일한 기본 경로 사용
    out_dir = args.out_dir or os.path.join(ckpt_dir, ckpt_dirname)
    os.makedirs(out_dir, exist_ok=True)

    # 손실 파라미터 검증
    if args.beta is None:
        print("Error: --beta 값을 정의해주세요.")
        sys.exit(1)
    masks = (args.loss == "hybrid")
    if masks and args.gamma is None:
        print("Error: hybrid 모드에서는 --gamma 값을 정의해주세요.")
        sys.exit(1)

    # 모델 및 체크포인트 로드
    model = ExplainerClassifierCNN(
        num_classes=args.nr_classes,
        img_size=tuple(args.img_size),
        clf=args.classifier,
        init_bias=args.init_bias,
    )
    ckpt = load_checkpoint_safely(ckpt_path, device)
    if isinstance(ckpt, dict) and "classifier" in ckpt and "explainer" in ckpt:
        model.classifier.load_state_dict(ckpt["classifier"])
        model.explainer.load_state_dict(ckpt["explainer"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)

    # 데이터 로드
    tr_df, val_df, test_df, _, classes = load_data(
        folder=args.dataset_path,
        dataset=args.dataset,
        masks=masks,
        class_weights=None
    )
    if test_df is None:
        test_df = val_df
    classes = [str(c) for c in classes]

    # DataLoader
    test_dataset = Dataset(
        test_df, preprocess=utils.norm, masks=masks,
        img_size=tuple(args.img_size), aug_prob=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    print(f"[INFO] Test images: {len(test_dataset)}  (batches: {len(test_loader)}, batch_size={args.batch_size})")

    # 평가
    test_global_loss, test_exp_loss, test_cls_loss, test_acc, probs, _, labels = \
        model.test(test_loader, device, args, args.alpha)

    # ROC/PR 커브 저장 (out_dir에 저장)
    if args.nr_classes > 2:
        utils.plot_roc_curve_multiclass(out_dir, probs, labels, classes)
        utils.plot_precision_recall_curve_multiclass(out_dir, probs, labels, classes)
    else:
        utils.plot_roc_curve(out_dir, probs, labels)
        utils.plot_precision_recall_curve(out_dir, probs, labels)

    # 결과 출력
    print(f"Test Loss : {test_global_loss:.4f}")
    print(f"Exp  Loss : {test_exp_loss:.4f}")
    print(f"Cls  Loss : {test_cls_loss:.4f}")
    print(f"Accuracy  : {test_acc:.4f}")

    # 설명맵 저장
    model.save_explanations(
        test_loader, 2, device, out_dir,
        test=True, classes=classes, only_explanations=True, cmap=args.cmap
    )


if __name__ == "__main__":
    main()
