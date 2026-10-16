r"""
train_bbb_leadcount_seeded.py — patch_seed.py 가 자동 생성.
직접 고치지 마십시오. 원본을 고친 뒤 패처를 다시 돌리십시오.

원본과의 유일한 차이는 --seed 와 --deterministic 입니다.
--seed 를 주지 않으면 원본과 동일하게 동작합니다.
"""
"""
train_bbb_leadcount.py

Chain #20 (JBI-rejected "Quantifying Performance Disparity and Restoring
Sensitivity via Subgroup Loss-Reweighting in Deep Learning-Based ECG
Diagnosis") 논문을 IEEE J-BHI에 재투고하기 위해, 기존 12-lead 단일 조건
실험을 1-lead / 3-lead / 12-lead 3개 조건으로 확장하는 스크립트입니다.

기존 파이프라인과의 관계
------------------------
- Dataset/모델 로딩 방식은 evaluate_debias.py의 MemmapECGDataset +
  resnet1d.py의 ResNet1D34를 그대로 따릅니다. 이 조합은 X.npy를
  (5000, 12) 그대로 읽어서 모델에 넘기고, ResNet1D34.forward가 내부에서
  x.permute(0,2,1)로 (B,T,C)->(B,C,T)를 직접 처리하기 때문에,
  train_naft.py 계열(NAFTResNet1D, permute 없음)에서 발견된 축-정렬
  버그가 여기에는 없습니다 -- 즉 기존 논문의 Table 1 수치(Vanilla/Debiased,
  12-lead)는 그대로 유효하며 다시 돌릴 필요가 없습니다.
- 이 스크립트가 새로 하는 일은 오직 하나, 리드 개수를 1/3/12로 바꿔가며
  같은 실험(Vanilla vs. Class-Group Weighted Loss)을 반복하는 것입니다.
- 데이터/분할 경로는 논문 Appendix A.1에 적힌 수치(Train 4,384 / Val 2,715
  / Test 5,429)와 정확히 일치하는 걸 확인한
  VitalDBK-MIMIC/preprocessed_27178_v2/imbalanced_split 을 그대로 재사용합니다.
- 손실함수는 논문 Section 2.3의 정의를 그대로 구현한 것입니다:
      L = (1/B) * sum( w_{g(i)} * CE(f(x_i), y_i) )
      w_g = N_total / (K * N_g)   (그룹별 고정 가중치, DRO 아님)
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from resnet1d import ResNet1D34

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_SETS = {
    "1lead": [1],          # Lead II -- 기존 논문과 동일 조건
    "3lead": [0, 1, 10],   # I, II, V5 -- AFib 리드카운트 실험과 동일한 웨어러블 3-패치 구성
    "12lead": list(range(12)),
}
GROUP_MAP = {'Asian': 0, 'Black': 1, 'Hispanic/Latino': 2, 'White': 3, 'Other': 4}
GROUP_NAMES = ['Asian', 'Black', 'Hispanic/Latino', 'White', 'Other']


class BBBLeadDataset(Dataset):
    """X.npy(N,5000,12)에서 lead_indices로 지정된 채널만 뽑아 (5000, C)로 반환.
    ResNet1D34.forward가 내부에서 permute(0,2,1)을 수행하므로 여기서는
    transpose 하지 않고 (T, C) 형태 그대로 반환한다."""

    def __init__(self, x_path, y_path, groups_path, indices, lead_indices):
        self.x_path = x_path
        self.y_path = y_path
        self.groups_path = groups_path
        self.indices = indices
        self.lead_indices = lead_indices
        self.X_mmap = None
        self.y_mmap = None
        self.groups = None

    def _ensure_open(self):
        if self.X_mmap is None:
            self.X_mmap = np.load(self.x_path, mmap_mode='r')
            self.y_mmap = np.load(self.y_path, mmap_mode='r')
            self.groups = np.load(self.groups_path, allow_pickle=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()
        real_idx = self.indices[idx]
        window = np.array(self.X_mmap[real_idx], dtype=np.float32)  # (5000, 12)
        x = window[:, self.lead_indices]  # (5000, C)
        y_data = int(self.y_mmap[real_idx])
        group_str = str(self.groups[real_idx])
        group_id = GROUP_MAP.get(group_str, 4)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_data, dtype=torch.long),
            torch.tensor(group_id, dtype=torch.long),
        )


class GroupWeightedCELoss(nn.Module):
    """논문 Section 2.3의 Class-Group Weighted Loss.
    w_g = N_total / (K * N_g), 학습 내내 고정(DRO 아님)."""

    def __init__(self, group_weights):
        super().__init__()
        self.register_buffer('w', torch.as_tensor(group_weights, dtype=torch.float32))

    def forward(self, logits, targets, groups):
        ce = F.cross_entropy(logits, targets, reduction='none')
        w = self.w[groups]
        return (ce * w).mean()


def compute_group_weights(groups_encoded, num_groups=5):
    counts = np.bincount(groups_encoded, minlength=num_groups).astype(np.float64)
    n_total = counts.sum()
    weights = np.zeros(num_groups, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = n_total / (num_groups * counts[nonzero])
    return weights


def encode_groups(groups_arr):
    return np.array([GROUP_MAP.get(str(g), 4) for g in groups_arr], dtype=np.int64)


def delong_roc_variance(ground_truth, predictions):
    m = np.sum(ground_truth == 1)
    n = np.sum(ground_truth == 0)
    if m == 0 or n == 0:
        return np.nan, np.nan
    pos_preds = predictions[ground_truth == 1]
    neg_preds = predictions[ground_truth == 0]
    V10 = np.array([np.mean(pos > neg_preds) + 0.5 * np.mean(pos == neg_preds) for pos in pos_preds])
    V01 = np.array([np.mean(pos_preds > neg) + 0.5 * np.mean(pos_preds == neg) for neg in neg_preds])
    s10 = np.var(V10, ddof=1) if len(V10) > 1 else 0
    s01 = np.var(V01, ddof=1) if len(V01) > 1 else 0
    var = (s10 / m) + (s01 / n)
    auc = roc_auc_score(ground_truth, predictions)
    return auc, var


def calc_sens_spec(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec


def evaluate(model, loader, device, groups_full, indices):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(yb.numpy())
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    test_groups = groups_full[indices]

    overall_auc, overall_var = delong_roc_variance(all_targets, all_probs)
    print(f"  전체 AUC = {overall_auc:.4f} (N={len(indices)})")

    for gname in GROUP_NAMES:
        mask = test_groups == gname
        if mask.sum() == 0:
            continue
        y_sub = all_targets[mask]
        p_sub = all_probs[mask]
        if len(np.unique(y_sub)) < 2:
            print(f"    {gname:>16s}  N={mask.sum():5d}  (양성/음성 한쪽만 존재, AUC 계산 불가)")
            continue
        auc, _ = delong_roc_variance(y_sub, p_sub)
        sens, spec = calc_sens_spec(y_sub, p_sub)
        print(f"    {gname:>16s}  N={mask.sum():5d}  AUC={auc:.4f}  Sensitivity={sens:.4f}  Specificity={spec:.4f}")
    return overall_auc, all_probs, all_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--lead_set", type=str, required=True, choices=list(LEAD_SETS.keys()))
    parser.add_argument("--mode", type=str, required=True, choices=["vanilla", "debiased"])
    parser.add_argument("--out_dir", type=str, default="./model_bbb_leadcount")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; omit to reproduce the original "
                             "unseeded behaviour of the six published checkpoints")
    parser.add_argument("--deterministic", action="store_true",
                        help="also force deterministic cuDNN kernels (slower, "
                             "and some ops may raise)")
    args = parser.parse_args()

    # ---- seeding -------------------------------------------------------
    # Only the seed changes between runs of this sweep. Data loading, the
    # fixed split arrays, the loss, the architecture and the optimiser are
    # untouched, so between-seed variation is attributable to initialisation
    # and batch ordering alone.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"[SEED] seed={args.seed} deterministic={args.deterministic}")
    else:
        print("[SEED] no seed set (original behaviour)")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lead_indices = LEAD_SETS[args.lead_set]
    lead_names = [LEAD_ORDER[i] for i in lead_indices]
    print(f"[INFO] mode={args.mode} | lead_set={args.lead_set} ({lead_names}) | device={device} | epochs={args.epochs}")

    x_path = os.path.join(args.data_dir, "X.npy")
    y_path = os.path.join(args.data_dir, "y.npy")
    groups_path = os.path.join(args.data_dir, "groups.npy")
    groups_full = np.load(groups_path, allow_pickle=True)

    train_idx = np.load(os.path.join(args.split_dir, "train_imbalanced_idx.npy"))
    val_idx = np.load(os.path.join(args.split_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(args.split_dir, "test_idx.npy"))
    print(f"[INFO] train N={len(train_idx)}, val N={len(val_idx)}, test N={len(test_idx)}")

    train_ds = BBBLeadDataset(x_path, y_path, groups_path, train_idx, lead_indices)
    val_ds = BBBLeadDataset(x_path, y_path, groups_path, val_idx, lead_indices)
    test_ds = BBBLeadDataset(x_path, y_path, groups_path, test_idx, lead_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ResNet1D34(in_channels=len(lead_indices), num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "debiased":
        train_groups_encoded = encode_groups(groups_full[train_idx])
        group_weights = compute_group_weights(train_groups_encoded, num_groups=5)
        print("[INFO] 고정 group weight (Class-Group Weighted Loss, Section 2.3):")
        for gi, gname in enumerate(GROUP_NAMES):
            print(f"    {gname:>16s}  w={group_weights[gi]:.4f}")
        criterion = GroupWeightedCELoss(group_weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    tag = f"{args.mode}_{args.lead_set}"
    best_model_path = os.path.join(args.out_dir, f"model_{tag}.pt")
    val_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, gb in train_loader:
            xb, yb, gb = xb.to(device), yb.to(device), gb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            if args.mode == "debiased":
                loss = criterion(logits, yb, gb)
            else:
                loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += val_criterion(logits, yb).item() * len(yb)
        val_loss /= len(val_ds)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"\n[SUCCESS] 저장 완료: {best_model_path} (Best Val Loss: {best_val_loss:.4f})")

    # 최종 리포트는 논문과 동일하게 고정 held-out test set(N=5,429)으로 평가
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\n=== [mode={args.mode}, lead_set={args.lead_set}] 최종 test 결과 (N={len(test_idx)}) ===")
    evaluate(model, test_loader, device, groups_full, test_idx)


if __name__ == "__main__":
    main()
