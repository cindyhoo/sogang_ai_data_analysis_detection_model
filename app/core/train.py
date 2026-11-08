"""학습 Job 엔트리 (제안서 §5.3): python -m app.core.train --job-dir <dir>

trainer 상주 프로세스가 Job마다 이 모듈을 자식 프로세스로 실행한다.
Job이 끝나면 프로세스가 종료되면서 TensorFlow·GPU 메모리가 OS로 돌아간다.
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time
import traceback
from pathlib import Path

from app.core import config, registry
from app.core.architectures import BN_MOMENTUM
from app.core.datasets import get_dataset, split_train_val
from app.core.fsutil import append_jsonl, now_iso, read_json, timestamp_id
from app.core.i18n import error_fields, localized
from app.core.jobs import cancel_requested, write_result, write_runtime
from app.core.runtime import env_info, setup_device

DEFAULTS = dict(
    name="", dataset="cifar100_fine", arch="xception_mini", width=1.0, epochs=25, batch_size=128,
    lr=3e-4, dropout=0.25, augment=["flip", "rotation"], val_split=0.1, train_subset=1.0,
    patience=5, seed=1337, device="auto",
)

STOP = {"flag": False}


def _on_sigterm(signum, frame):     # trainer 종료·취소 유예시간 초과 시 SIGTERM
    STOP["flag"] = True


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def make_augment(names):
    from keras import layers

    aug = []
    if "flip" in names:
        aug.append(layers.RandomFlip("horizontal"))
    if "rotation" in names:
        aug.append(layers.RandomRotation(0.1))
    if "translation" in names:
        aug.append(layers.RandomTranslation(0.1, 0.1))
    if not aug:
        return None

    def augment(images):
        for layer in aug:
            images = layer(images)
        return images

    return augment


def make_dataset(x, y, batch_size, shuffle, augment, seed):
    """uint8 그대로 두고 배치 단위로 float 변환 → 증강 (예제의 dataset.map 방식)."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 10_000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    if augment is not None:
        ds = ds.map(lambda a, b: (augment(a), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_progress_callback(job_dir: Path, epochs: int, steps: int):
    import keras

    class ProgressCallback(keras.callbacks.Callback):
        """진행률을 progress.jsonl에 기록하고, 취소 플래그를 보면 학습을 멈춘다."""

        def __init__(self):
            super().__init__()
            self.records: list[dict] = []
            self.cancelled = False
            self.every = max(1, steps // 20)
            self.path = job_dir / "progress.jsonl"

        def on_train_begin(self, logs=None):
            self.t0 = time.time()

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            self.ep_t0 = time.time()

        def on_train_batch_end(self, batch, logs=None):
            if STOP["flag"] or cancel_requested(job_dir):
                self.cancelled = True
                self.model.stop_training = True
            if batch % self.every == 0 or batch == steps - 1:
                logs = logs or {}
                append_jsonl(self.path, {
                    "type": "batch", "epoch": self.epoch + 1, "epochs": epochs,
                    "batch": batch + 1, "steps": steps,
                    "loss": _f(logs.get("loss")), "acc": _f(logs.get("acc")),
                    "elapsed": round(time.time() - self.t0, 1),
                })

        def on_epoch_end(self, epoch, logs=None):
            done = epoch + 1
            elapsed = time.time() - self.t0
            rec = {"type": "epoch", "epoch": done, "epochs": epochs}
            rec.update({k: _f(v) for k, v in (logs or {}).items()})
            rec["epoch_time"] = round(time.time() - self.ep_t0, 2)
            rec["elapsed"] = round(elapsed, 1)
            rec["eta_sec"] = round(elapsed / done * (epochs - done), 1)
            self.records.append(rec)
            append_jsonl(self.path, rec)
            print(f"[epoch {done}/{epochs}] " + " ".join(
                f"{k}={v:.4f}" for k, v in rec.items() if k in ("loss", "acc", "top5", "val_loss", "val_acc", "val_top5") and v is not None
            ) + f" ({rec['epoch_time']:.1f}s)", flush=True)

    return ProgressCallback()


def run(job_dir: Path) -> int:
    params = {**DEFAULTS, **(read_json(job_dir / "params.json", {}) or {})}
    model_id = None
    meta = None
    try:
        write_runtime(job_dir, **localized("phase", "phase.device"))
        pref = params["device"] if params["device"] != "auto" else config.TRAINER_DEVICE
        device = setup_device(pref)
        print(f"training device: {device}", flush=True)

        import keras

        seed = int(params["seed"])
        keras.utils.set_random_seed(seed)
        spec = get_dataset(params["dataset"])

        write_runtime(job_dir, **localized("phase", "phase.loading_data"), device=device)
        (x, y), _ = spec.load()          # test set은 학습 중에 쓰지 않는다
        x_tr, y_tr, x_val, y_val = split_train_val(
            x, y, float(params["val_split"]), seed, float(params["train_subset"])
        )
        del x, y
        batch_size = int(params["batch_size"])
        epochs = int(params["epochs"])
        steps = math.ceil(len(x_tr) / batch_size)
        augment = make_augment(params["augment"])
        train_ds = make_dataset(x_tr, y_tr, batch_size, True, augment, seed)
        val_ds = make_dataset(x_val, y_val, batch_size, False, None, seed)
        print(f"data: train={len(x_tr)} val={len(x_val)} steps/epoch={steps}", flush=True)

        width = float(params["width"])
        model_id = registry.new_model_id(timestamp_id(), params["arch"], width, spec.id)
        mdir = registry.model_dir(model_id)
        mdir.mkdir(parents=True)
        meta = {
            "id": model_id,
            "name": params["name"] or f"{params['arch']} w{registry.width_tag(width)} ({spec.label_mode})",
            "status": "training",
            "dataset": spec.id,
            "num_classes": spec.num_classes,
            "class_names": list(spec.class_names),
            "input_shape": list(spec.input_shape),
            "arch": params["arch"],
            "hparams": {k: params[k] for k in DEFAULTS if k not in ("name", "dataset", "arch", "device")},
            "arch_config": {"bn_momentum": BN_MOMENTUM},
            "data": {"train": len(x_tr), "val": len(x_val)},
            "device": device,
            "train_env": env_info(config.TRAINER_ENV),
            "job_id": job_dir.name,
            "num_params": None,
            "best": None,
            "epochs_completed": 0,
            "train_time_sec": None,
            "created_at": now_iso(),
            "finished_at": None,
            "tags": [],
            "memo": "",
        }
        registry.write_meta(model_id, meta)
        write_runtime(job_dir, **localized("phase", "phase.training"), model_id=model_id, steps=steps, epochs=epochs,
                      train_size=len(x_tr), val_size=len(x_val))

        from app.core.architectures import build_model

        model = build_model(params["arch"], spec.input_shape, spec.num_classes, width, float(params["dropout"]))
        model.compile(
            optimizer=keras.optimizers.Adam(float(params["lr"])),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="acc"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
            ],
        )
        lines: list[str] = []
        model.summary(print_fn=lambda s, **kw: lines.append(s), line_length=100)
        (mdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
        meta["num_params"] = int(model.count_params())
        registry.write_meta(model_id, meta)

        progress = make_progress_callback(job_dir, epochs, steps)
        callbacks = [
            keras.callbacks.ModelCheckpoint(str(mdir / "model.keras"), monitor="val_acc", mode="max",
                                            save_best_only=True),
            keras.callbacks.CSVLogger(str(mdir / "history.csv")),
            progress,
        ]
        early = None
        if int(params["patience"]) > 0:
            early = keras.callbacks.EarlyStopping(monitor="val_acc", mode="max",
                                                  patience=int(params["patience"]), restore_best_weights=True)
            callbacks.insert(1, early)

        t0 = time.time()
        model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=2)
        train_time = time.time() - t0

        cancelled = progress.cancelled or STOP["flag"]
        recs = progress.records
        best = max(recs, key=lambda r: r.get("val_acc") or -1) if recs else None
        has_ckpt = (mdir / "model.keras").exists()
        meta.update(
            status="cancelled" if cancelled else ("ready" if has_ckpt else "failed"),
            best={k: best.get(k) for k in ("epoch", "val_acc", "val_top5", "val_loss", "acc", "loss")} if best else None,
            epochs_completed=len(recs),
            early_stopped=bool(early and early.stopped_epoch > 0),
            train_time_sec=round(train_time, 1),
            avg_epoch_sec=round(sum(r["epoch_time"] for r in recs) / len(recs), 2) if recs else None,
            finished_at=now_iso(),
        )
        registry.write_meta(model_id, meta)
        outcome = "cancelled" if cancelled else "succeeded"
        write_result(job_dir, outcome=outcome, model_id=model_id, best=meta["best"])
        print(f"done: {outcome} model_id={model_id} train_time={train_time:.1f}s", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001 - 실패 원인을 상태 파일에 남긴다
        traceback.print_exc()
        if model_id and meta is not None:
            meta.update(status="failed", **error_fields("error", e), finished_at=now_iso())
            registry.write_meta(model_id, meta)
        write_result(job_dir, outcome="failed", **error_fields("error", e), model_id=model_id)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _on_sigterm)
    return run(Path(args.job_dir))


if __name__ == "__main__":
    sys.exit(main())
