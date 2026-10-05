import os
import sys
import json
import time
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils import list_images, CLASS_NAMES
from pca import StandardScalerFromScratch, PCAFromScratch
from models import LinearRegressionFromScratch, LogisticRegressionFromScratch
from svm import LinearSVMFromScratch
from metrics import evaluate_binary, classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "PeopleCountingDataSet")
OUT_DIR = os.path.join(BASE, "models")
RES_DIR = os.path.join(BASE, "results")

IMG_SIZE = 48
N_COMPONENTS = 100
VAL_RATIO = 0.2
SEED = 42


def to_binary(y):
    return (np.asarray(y) > 0).astype(np.int64)


def load_dataset(img_size=IMG_SIZE):
    X, y3, paths = [], [], []
    for ci, cn in enumerate(CLASS_NAMES):
        files = list_images(ROOT, cn)
        for i, f in enumerate(files):
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            g = cv2.resize(g, (img_size, img_size), interpolation=cv2.INTER_AREA)
            X.append(g.astype(np.float32) / 255.0)
            y3.append(ci)
            paths.append(f)
            if (i + 1) % 5000 == 0:
                print(f"    [{cn}] {i + 1}/{len(files)}")
    return np.asarray(X, np.float32), np.asarray(y3), np.asarray(paths)


def grad_feats(A, img_size=IMG_SIZE):
    imgs = A.reshape(-1, img_size, img_size)
    gx = np.abs(np.diff(imgs, axis=2, prepend=imgs[:, :, :1]))
    gy = np.abs(np.diff(imgs, axis=1, prepend=imgs[:, :1, :]))
    return (gx + gy).reshape(len(A), -1)


def make_features(A, bg, img_size=IMG_SIZE):
    raw = A.reshape(len(A), -1)
    diff = np.abs(raw - bg.reshape(-1))
    return np.hstack([raw, grad_feats(A, img_size), diff])


def stratified_split(y, val_ratio, seed):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = int(len(idx) * val_ratio)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 64)
    print("PERSON DETECTION - TRAINING PIPELINE")
    print("=" * 64)
    print(f"[1] Doc dataset: {ROOT}  (img_size={IMG_SIZE})")
    X, y3, paths = load_dataset()
    print(f"    Tong anh: {len(X)}  feature raw dim = {X.shape[1] * X.shape[1]}")
    print(f"    Phan bo 3 lop: {np.bincount(y3)}")
    y = to_binary(y3)
    print(f"    Phan bo nhi phan (0=khong nguoi, 1=co nguoi): {np.bincount(y)}")

    tr_idx, va_idx = stratified_split(y, VAL_RATIO, SEED)
    Xtr, ytr, y3tr = X[tr_idx], y[tr_idx], y3[tr_idx]
    Xva, yva, y3va = X[va_idx], y[va_idx], y3[va_idx]
    print(f"[2] Chia tap: train={len(Xtr)}  test={len(Xva)}")

    bg = Xtr[ytr == 0].mean(axis=0)
    print("[3] Tao dac trung: raw + gradient + background-diff...")
    Ftr = make_features(Xtr, bg)
    Fva = make_features(Xva, bg)
    print(f"    feature dim = {Ftr.shape[1]}")

    print("[4] Chuan hoa z-score...")
    scaler = StandardScalerFromScratch().fit(Ftr)
    Ftr_s, Fva_s = scaler.transform(Ftr), scaler.transform(Fva)

    print(f"[5] PCA giam chieu (eigen decomposition, {N_COMPONENTS} thanh phan)...")
    pca = PCAFromScratch(n_components=N_COMPONENTS).fit(Ftr_s)
    Ztr, Zva = pca.transform(Ftr_s), pca.transform(Fva_s)
    evr = pca.explained_variance_ratio_
    print(f"    Giu lai {evr.sum() * 100:.2f}% phuong sai")

    print("[6] Chuan hoa dau ra PCA...")
    pca_scaler = StandardScalerFromScratch().fit(Ztr)
    Ztr, Zva = pca_scaler.transform(Ztr), pca_scaler.transform(Zva)

    np.savez(os.path.join(OUT_DIR, "preprocess.npz"),
             scaler_mean=scaler.mean_, scaler_std=scaler.std_,
             pca_mean=pca.mean_, pca_components=pca.components_,
             pca_scaler_mean=pca_scaler.mean_, pca_scaler_std=pca_scaler.std_,
             background=bg.reshape(-1), img_size=IMG_SIZE)

    results = {}
    models = {}

    print("[7] Train Linear Regression...")
    lin = LinearRegressionFromScratch(learning_rate=0.005, n_iters=5000, reg=1e-4).fit(Ztr, ytr)
    results["linear_regression"] = evaluate_binary(yva, lin.predict(Zva), lin.predict_scores(Zva))
    models["linear_regression"] = lin
    r = results["linear_regression"]
    print(f"    acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['roc_auc']:.4f}")

    print("[8] Train Logistic Regression...")
    log = LogisticRegressionFromScratch(learning_rate=1.0, n_iters=6000, reg=1e-4).fit(Ztr, ytr)
    results["logistic_regression"] = evaluate_binary(yva, log.predict(Zva), log.predict_proba(Zva))
    models["logistic_regression"] = log
    r = results["logistic_regression"]
    print(f"    acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['roc_auc']:.4f}")

    print("[9] Train Linear SVM (hinge loss + L2)...")
    svm = LinearSVMFromScratch(learning_rate=0.001, n_iters=5000, C=1.0, batch_size=256).fit(Ztr, ytr)
    results["svm"] = evaluate_binary(yva, svm.predict(Zva), svm.decision_function(Zva))
    models["svm"] = svm
    r = results["svm"]
    print(f"    acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['roc_auc']:.4f}")

    print("[10] Luu model...")
    for name, m in models.items():
        np.savez(os.path.join(OUT_DIR, f"{name}.npz"), w=m.w, b=np.array([m.b]))

    print("[11] Danh gia phan lop 3 lop (One-vs-Rest, Logistic Regression)...")
    ovr = train_ovr(Ztr, y3tr, Zva, y3va)

    with open(os.path.join(RES_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "config": {"img_size": IMG_SIZE, "n_components": N_COMPONENTS,
                       "feature_dim": int(Ftr.shape[1]),
                       "pca_explained_variance": float(evr.sum()),
                       "n_train": int(len(Xtr)), "n_test": int(len(Xva)),
                       "class_names": CLASS_NAMES},
            "binary_results": results,
            "three_class": ovr,
        }, f, indent=2)

    _plot_curves(models, pca)
    print(f"[OK] Hoan tat trong {time.time() - t0:.1f}s")


def train_ovr(Ztr, y3tr, Zva, y3va):
    models = []
    for c in range(3):
        yc = (y3tr == c).astype(np.int64)
        m = LogisticRegressionFromScratch(learning_rate=1.0, n_iters=6000, reg=1e-4).fit(Ztr, yc)
        models.append(m)
    scores = np.column_stack([m.predict_proba(Zva) for m in models])
    pred = scores.argmax(axis=1)
    rep = classification_report(y3va, pred, 3, CLASS_NAMES)
    print(f"    acc(3 lop)={rep['accuracy']:.4f}  macro_f1={rep['macro_f1']:.4f}")
    return rep


def _plot_curves(models, pca):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, m in models.items():
        axes[0].plot(m.loss_history_, label=name)
    axes[0].set_title("Loss theo vong lap")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(np.cumsum(pca.explained_variance_ratio_), marker="o", ms=3)
    axes[1].set_title("PCA - phuong sai tich luy")
    axes[1].set_xlabel("so thanh phan")
    axes[1].set_ylabel("explained variance ratio")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, "training_curves.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
