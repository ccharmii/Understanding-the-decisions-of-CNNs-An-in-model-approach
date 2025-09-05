# calc_metrics_new.py  (SCOUTER 동치 최종판, ExplainerClassifierCNN에 맞춤)

import os, glob, argparse, inspect
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from Dataset import load_data, Dataset
from ExplainerClassifierCNN import ExplainerClassifierCNN
import utils

# ===== 고정 설정(원 SCOUTER/RISE) =====
IMG_SIZE = 224
STEP = IMG_SIZE                      # 한 스텝당 픽셀 수 (SCOUTER: 224)
HW = IMG_SIZE * IMG_SIZE
K = (HW + STEP - 1) // STEP          # 224
SEED = 0
INFID_N = 50
SEN_R   = 0.2
SEN_N   = 50
EPS     = 1e-8
# ====================================

# ---------- PyTorch 2.6 호환: 안전 체크포인트 로더 ----------
def load_checkpoint_safely(path, device):
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
# ------------------------------------------------------------

def set_seeds(s=SEED):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def gkern(klen=11, sigma=5):
    """3채널 동일 가우시안 커널(Conv2d용)"""
    base = np.zeros((klen, klen), dtype=np.float32); base[klen//2, klen//2] = 1.0
    k = gaussian_filter(base, sigma)
    ker = np.zeros((3, 3, klen, klen), dtype=np.float32)
    for ch in range(3): ker[ch, ch] = k
    return torch.from_numpy(ker)

def load_explanation(path):
    """설명맵 PNG → (H,W) uint8 (그레이스케일로 강제 변환: SCOUTER 정합)"""
    return np.array(Image.open(path).convert('L').resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)

def _auc(arr: np.ndarray) -> float:
    """[0,1] 구간 정규화된 trapezoid AUC"""
    arr = np.asarray(arr, dtype=np.float64)
    T = len(arr) - 1
    if T <= 0: return 0.0
    return (arr.sum() - 0.5*(arr[0]+arr[-1])) / T

@torch.no_grad()
def _target_class_from_original(model, img):
    """원본 이미지에서 타깃 클래스 고정 (SCOUTER/RISE 방식)"""
    prob = model(img, softmax=True)   # (1,C)
    return int(torch.argmax(prob, dim=1).item())

@torch.no_grad()
def _causal_auc_scout(model, img_orig, start, finish, exp_u8):
    """
    IAUC/DAUC 공통:
      - 타깃 클래스: 원본 이미지에서 고정
      - 각 스텝: softmax 확률의 해당 클래스 점수 적분
    """
    c = _target_class_from_original(model, img_orig)
    order = np.argsort(-exp_u8.reshape(-1))  # 내림차순

    x = start.clone()
    scores = []
    for i in range(K+1):
        prob = model(x, softmax=True)[0, c].item()
        scores.append(prob)
        if i < K:
            idx = order[i*STEP:(i+1)*STEP]
            r, col = np.unravel_index(idx, (IMG_SIZE, IMG_SIZE))
            x[0, :, r, col] = finish[0, :, r, col]
    return _auc(np.asarray(scores, np.float64))

def compute_iauc(model, img, exp_u8):
    ker = gkern().to(next(model.parameters()).device)
    blur = lambda t: F.conv2d(t, ker, padding=ker.shape[-1]//2)
    return _causal_auc_scout(model, img, blur(img), img.clone(), exp_u8)

def compute_dauc(model, img, exp_u8):
    return _causal_auc_scout(model, img, img.clone(), torch.zeros_like(img), exp_u8)

# ---------- SCOUTER 정의와 동치: Infidelity (logit 사용) ----------
@torch.no_grad()
def compute_infidelity(model, img, exp_u8, label):
    """
    Yeh et al. / SCOUTER 적응:
      - 모델 출력: softmax 없이 '원 출력(=logit)' 사용 (SCOUTER 코드와 동일)
      - β-스케일링 포함
      - 입력 픽셀 감소형 가우시안 변형
    """
    set_seeds(SEED)
    H = W = IMG_SIZE; HW = H*W; point = HW

    # p0: logit
    p0 = model(img)[0, int(label)].item()

    exp = np.asarray(exp_u8, dtype=np.float32).reshape(-1)
    ks_all, lhs_all, rhs_all = [], [], []

    for _ in range(INFID_N):
        ind  = np.random.choice(HW, point, replace=False)
        rand = np.random.normal(0.0, 0.2, size=point).astype(np.float32)

        exp_sum = float((rand * exp[ind]).sum()); ks = 1.0

        xi = img.clone()
        r, c = np.unravel_index(ind, (H, W))
        rand_t = torch.from_numpy(rand).to(xi.device)
        for ch in range(xi.shape[1]):
            xi[0, ch, r, c] = xi[0, ch, r, c] - rand_t

        # p1: logit
        p1 = model(xi)[0, int(label)].item()

        ks_all.append(ks); lhs_all.append(ks*exp_sum); rhs_all.append(ks*(p0 - p1))

    ks_all = np.asarray(ks_all, np.float32)
    lhs_all = np.asarray(lhs_all, np.float32)
    rhs_all = np.asarray(rhs_all, np.float32)

    denom = (ks_all * lhs_all * lhs_all).mean()
    beta  = 0.0 if denom == 0 else (ks_all * rhs_all * lhs_all).mean() / denom
    lhs_all *= beta

    infid = ((rhs_all - lhs_all)**2 / np.maximum(ks_all, EPS)).mean()
    return float(infid)

# ---------- SCOUTER 정의와 동치: Sensitivity (||Δ|| / ||expl||) ----------
def _get_expl_from_model(model, x):
    with torch.no_grad():
        out = model(x)                      # (logits) 또는 (logits, expl)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        return out[1]
    # Explainer 모듈 직접 호출
    return model.explainer(x)

def _minmax_to_255(arr):
    arr = np.asarray(arr, dtype=np.float32)
    a, b = float(arr.min()), float(arr.max())
    if b > a:
        return ((arr - a)/(b-a)*255.0).astype(np.float32)
    return np.zeros_like(arr, np.float32)

@torch.no_grad()
def compute_sensitivity(model, img, exp_u8):
    """
    SCOUTER: sens = max(||expl - expl_eps||₂) / ||expl||₂  (둘 다 0..255)
    """
    set_seeds(SEED)
    base = np.asarray(exp_u8, dtype=np.float32)  # (H,W) 0..255
    base_norm = float(np.linalg.norm(base)) + EPS

    max_diff = 0.0
    for _ in range(SEN_N):
        noise = np.random.uniform(-SEN_R, SEN_R, size=img.shape).astype(np.float32)
        xi = img + torch.from_numpy(noise).to(img.device)

        expl = _get_expl_from_model(model, xi).squeeze().detach().cpu().numpy().astype(np.float32)
        expl = _minmax_to_255(expl)  # 0..255 정합

        diff = float(np.linalg.norm(base - expl))
        if diff > max_diff: max_diff = diff

    sens = max_diff / base_norm
    return float(sens)

def _find_exp_file(exp_dir, basename):
    """
    a 모델 저장 규칙:
      exp_dir/**/{timestamp}_phase{phase}_only_expl_{basename}
    를 재귀 검색.
    """
    pat = os.path.join(exp_dir, "**", f"*phase2*only_expl*{basename}")
    files = glob.glob(pat, recursive=True)
    if files: return files[0]
    # fallback: phase 미포함 패턴도 검색
    pat2 = os.path.join(exp_dir, "**", f"*only_expl*{basename}")
    files2 = glob.glob(pat2, recursive=True)
    return files2[0] if files2 else None

def main(args):
    # PyTorch 기본 train()로 덮어쓰기(원본 호환)
    ExplainerClassifierCNN.train = torch.nn.Module.train
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    tr_df, val_df, test_df, _, classes = load_data(folder=args.dataset_path, dataset=args.dataset, masks=False)
    split_df = {"train": tr_df, "val": val_df, "test": test_df}[args.split]

    # DataLoader (augmentation off)
    ds = Dataset(split_df, preprocess=utils.norm, masks=False, img_size=tuple(args.img_size), aug_prob=0)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    loader = tqdm(loader, desc="Evaluating Images", unit="img")

    # 모델 로드 (안전 로더 사용)
    model = args.model_class(
        num_classes=len(classes),
        img_size=tuple(args.img_size),
        clf=args.classifier,
        init_bias=args.init_bias,
        pretrained=False
    ).to(device)

    ckpt = load_checkpoint_safely(args.model_path, device)
    if isinstance(ckpt, dict) and "classifier" in ckpt and "explainer" in ckpt:
        model.classifier.load_state_dict(ckpt["classifier"])
        model.explainer.load_state_dict(ckpt["explainer"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 평가
    records = []
    for img, label, img_path, _ in loader:
        img = img.to(device)
        basename = os.path.basename(img_path[0])

        exp_file = _find_exp_file(args.exp_dir, basename)
        if not exp_file:
            print(f"[WARN] 설명맵 없음: {basename}")
            continue
        exp_u8 = load_explanation(exp_file)

        iauc = compute_iauc(model, img, exp_u8)
        dauc = compute_dauc(model, img, exp_u8)
        infi = compute_infidelity(model, img, exp_u8, label.item())
        sens = compute_sensitivity(model, img, exp_u8)

        records.append({"image": basename, "iauc": iauc, "dauc": dauc,
                        "infidelity": infi, "sensitivity": sens})

    # 저장 및 요약
    df = pd.DataFrame(records)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print("\n=== 평균 지표 ===")
    print(df[["iauc","dauc","infidelity","sensitivity"]].mean().to_string())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--dataset", default="transfer")
    p.add_argument("--split", choices=["train","val","test"], default="val")
    p.add_argument("--model_path", required=True)
    p.add_argument("--exp_dir", required=True)             # run 폴더 또는 explanations 상위 폴더
    p.add_argument("--output_csv", default="metrics.csv")
    p.add_argument("--classifier", default="resnet50")
    p.add_argument("--init_bias", type=float, default=2.0)
    p.add_argument("--model_class", default="ExplainerClassifierCNN",
                   choices=["ExplainerClassifierCNN"])
    p.add_argument("--img_size", nargs=2, type=int, default=[224,224])
    args = p.parse_args()

    if args.model_class == "ExplainerClassifierCNN":
        args.model_class = ExplainerClassifierCNN

    main(args)
