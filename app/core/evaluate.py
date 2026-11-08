"""평가 Job 엔트리 (제안서 §7): python -m app.core.evaluate --job-dir <dir>

test set 10,000장을 한 번 예측하고 지표·그림·예측값을 evaluations/{eval_id}/에 저장한다.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from app.core import config, registry
from app.core.datasets import get_dataset
from app.core.fsutil import now_iso, read_json, timestamp_id, write_json_atomic
from app.core.i18n import AppError, error_fields, localized
from app.core.jobs import write_result, write_runtime
from app.core.runtime import setup_device
from app.core.stats import expected_calibration_error, softmax, wilson_ci

N_MISCLASSIFIED = 24


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_confusion_png(cm: np.ndarray, names: list[str], path: Path) -> None:
    plt = _plt()
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    n = len(names)
    fig, ax = plt.subplots(figsize=(9, 8) if n > 30 else (8.5, 7.5), dpi=110)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    if n <= 30:
        ax.set_xticks(range(n), names, rotation=90, fontsize=7)
        ax.set_yticks(range(n), names, fontsize=7)
    else:
        ticks = list(range(0, n, 10))
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (row-normalized = per-class recall)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_reliability_png(bins: list[dict], ece: float, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=110)
    width = bins[0]["hi"] - bins[0]["lo"]
    xs = [b["lo"] + width / 2 for b in bins if b["count"]]
    accs = [b["accuracy"] for b in bins if b["count"]]
    ax.bar(xs, accs, width=width * 0.92, color="#4f7cff", alpha=0.85, label="Accuracy")
    ax.plot([0, 1], [0, 1], "--", color="#888", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (max softmax probability)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(job_dir: Path) -> int:
    params = read_json(job_dir / "params.json", {}) or {}
    model_id = params.get("model_id")
    try:
        meta = registry.read_meta(model_id)
        if not meta or meta.get("status") != "ready":
            raise AppError("err.eval_needs_ready_id", id=model_id)
        pref = params.get("device") or "auto"
        device = setup_device(pref if pref != "auto" else config.TRAINER_DEVICE)

        import keras
        from PIL import Image
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

        spec = get_dataset(params.get("dataset") or meta["dataset"])
        if spec.num_classes != meta["num_classes"]:
            raise AppError("err.class_mismatch")
        names = list(spec.class_names)

        write_runtime(job_dir, **localized("phase", "phase.loading_data"), device=device, model_id=model_id)
        _, (x_te, y_te) = spec.load()
        write_runtime(job_dir, **localized("phase", "phase.loading_model"))
        model = keras.saving.load_model(str(registry.model_file(model_id)), compile=False)
        write_runtime(job_dir, **localized("phase", "phase.predicting", n=len(x_te)))
        t0 = time.time()
        logits = model.predict(x_te.astype("float32"), batch_size=256, verbose=0)
        pred_time = time.time() - t0

        probs = softmax(np.asarray(logits, dtype=np.float64))
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        correct = pred == y_te
        top5_idx = np.argpartition(-probs, kth=4, axis=1)[:, :5]
        top5_correct = (top5_idx == y_te[:, None]).any(axis=1)
        n = int(len(y_te))
        loss = float(-np.mean(np.log(probs[np.arange(n), y_te] + 1e-12)))

        labels = list(range(spec.num_classes))
        prec, rec, f1, support = precision_recall_fscore_support(y_te, pred, labels=labels, zero_division=0)
        cm = confusion_matrix(y_te, pred, labels=labels)
        per_class = [
            {"index": i, "name": names[i], "support": int(support[i]), "accuracy": float(rec[i]),
             "precision": float(prec[i]), "f1": float(f1[i])}
            for i in labels
        ]
        ranked = sorted(per_class, key=lambda r: (r["accuracy"], r["f1"]))
        off = cm.copy()
        np.fill_diagonal(off, 0)
        flat = np.argsort(off, axis=None)[::-1][:20]
        confused = []
        for k in flat:
            t, p = divmod(int(k), spec.num_classes)
            if off[t, p] == 0:
                break
            confused.append({"true": names[t], "pred": names[p], "count": int(off[t, p]),
                             "rate": float(off[t, p] / max(1, cm[t].sum()))})
        ece, bins = expected_calibration_error(conf, correct)

        eval_id = timestamp_id()
        edir = registry.evaluations_dir(model_id) / eval_id
        i = 2
        while edir.exists():
            eval_id = f"{timestamp_id()}-{i}"
            edir = registry.evaluations_dir(model_id) / eval_id
            i += 1
        (edir / "misclassified").mkdir(parents=True)
        write_runtime(job_dir, **localized("phase", "phase.metrics"), eval_id=eval_id)

        save_confusion_png(cm, names, edir / "confusion_matrix.png")
        save_reliability_png(bins, ece, edir / "reliability.png")
        with open(edir / "per_class.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
            w.writeheader()
            w.writerows(per_class)
        np.savez_compressed(edir / "preds.npz", y_true=y_te.astype(np.int16), y_pred=pred.astype(np.int16),
                            conf=conf.astype(np.float32))

        wrong = np.flatnonzero(~correct)
        wrong = wrong[np.argsort(-conf[wrong])][:N_MISCLASSIFIED]
        misclassified = []
        for j, idx in enumerate(wrong):
            fname = f"misclassified/{j:02d}.png"
            Image.fromarray(x_te[idx]).resize((96, 96), Image.Resampling.NEAREST).save(edir / fname)
            misclassified.append({"file": fname, "index": int(idx), "true": names[int(y_te[idx])],
                                  "pred": names[int(pred[idx])], "conf": float(conf[idx])})

        k1, k5 = int(correct.sum()), int(top5_correct.sum())
        metrics = {
            "eval_id": eval_id,
            "model_id": model_id,
            "dataset": spec.id,
            "split": "test",
            "n": n,
            "loss": loss,
            "top1": k1 / n,
            "top1_ci": list(wilson_ci(k1, n)),
            "top5": k5 / n,
            "top5_ci": list(wilson_ci(k5, n)),
            "macro_precision": float(np.mean(prec)),
            "macro_recall": float(np.mean(rec)),
            "macro_f1": float(np.mean(f1)),
            "ece": ece,
            "reliability_bins": bins,
            "per_class": per_class,
            "worst_classes": ranked[:10],
            "best_classes": ranked[::-1][:10],
            "confused_pairs": confused,
            "misclassified": misclassified,
            "files": {"confusion_matrix": "confusion_matrix.png", "reliability": "reliability.png",
                      "per_class_csv": "per_class.csv", "preds": "preds.npz"},
            "device": device,
            "predict_time_sec": round(pred_time, 2),
            "created_at": now_iso(),
            "job_id": job_dir.name,
        }
        write_json_atomic(edir / "metrics.json", metrics)
        write_result(job_dir, outcome="succeeded", model_id=model_id, eval_id=eval_id, top1=metrics["top1"])
        print(f"eval done: {model_id} {eval_id} top1={metrics['top1']:.4f} top5={metrics['top5']:.4f}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        write_result(job_dir, outcome="failed", **error_fields("error", e), model_id=model_id)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    return run(Path(parser.parse_args().job_dir))


if __name__ == "__main__":
    sys.exit(main())
