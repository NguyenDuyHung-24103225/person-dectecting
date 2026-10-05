import numpy as np


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(prec), float(rec), float(f1)


def roc_auc(y_true, scores):
    """Tinh ROC-AUC bang cong thuc rank (Mann-Whitney U)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = ranks[y_true == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def multiclass_confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def classification_report(y_true, y_pred, n_classes, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    cm = multiclass_confusion(y_true, y_pred, n_classes)
    report = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        report[class_names[c]] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(cm[c, :].sum()),
        }
    macro_prec = float(np.mean([report[n]["precision"] for n in class_names]))
    macro_rec = float(np.mean([report[n]["recall"] for n in class_names]))
    macro_f1 = float(np.mean([report[n]["f1"] for n in class_names]))
    return {
        "per_class": report,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "accuracy": float(np.trace(cm) / (cm.sum() + 1e-12)),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_binary(y_true, y_pred, scores=None):
    cm = confusion_matrix(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }
    if scores is not None:
        out["roc_auc"] = roc_auc(y_true, scores)
    return out
