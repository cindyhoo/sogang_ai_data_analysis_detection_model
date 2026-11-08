"""데이터셋 준비 Job 엔트리 (제안서 §4.4): python -m app.core.prepare --job-dir <dir>

1) keras.datasets.cifar100.load_data()로 내려받아 storage/keras/에 캐시
2) 클래스별 샘플 썸네일 생성 → storage/datasets/{id}/samples/{class:03d}.png
3) 클래스 분포·이미지 수·클래스 이름을 storage/datasets/{id}/info.json으로 저장
web은 이 결과물만 보여 주므로 원본 배열을 메모리에 올리지 않는다.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

from app.core import config
from app.core.datasets import get_dataset, info_path
from app.core.fsutil import now_iso, read_json, write_json_atomic
from app.core.i18n import error_fields, localized
from app.core.jobs import write_result, write_runtime

SAMPLES_PER_CLASS = 8


def check_meta_names(spec) -> bool | None:
    """내려받은 원본 meta 파일의 클래스 이름이 코드의 상수와 같은지 확인한다."""
    key = "fine_label_names" if spec.label_mode == "fine" else "coarse_label_names"
    for meta in config.KERAS_HOME.glob("datasets/**/cifar-100-python/meta"):
        try:
            with open(meta, "rb") as f:
                names = pickle.load(f, encoding="latin1")[key]
            return list(names) == list(spec.class_names)
        except Exception:  # noqa: BLE001
            continue
    return None


def run(job_dir: Path) -> int:
    params = read_json(job_dir / "params.json", {}) or {}
    try:
        from PIL import Image

        spec = get_dataset(params["dataset"])
        write_runtime(job_dir, **localized("phase", "phase.download"), device="CPU")
        (xtr, ytr), (xte, yte) = spec.load()

        out = config.DATASETS_DIR / spec.id
        samples = out / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        write_runtime(job_dir, **localized("phase", "phase.samples"))
        rng = np.random.default_rng(0)
        for c in range(spec.num_classes):
            idx = rng.choice(np.flatnonzero(ytr == c), SAMPLES_PER_CLASS, replace=False)
            strip = np.concatenate(list(xtr[idx]), axis=1)          # (32, 32*8, 3)
            Image.fromarray(strip).save(samples / f"{c:03d}.png")

        sub = xtr[rng.choice(len(xtr), 10_000, replace=False)].reshape(-1, 3).astype(np.float32)
        verified = check_meta_names(spec)
        if verified is False:
            print("WARNING: class names in the original meta file differ from the code constants.", flush=True)
        info = {
            **spec.to_dict(),
            "class_names": list(spec.class_names),
            "train_size": int(len(xtr)),
            "test_size": int(len(xte)),
            "class_counts_train": np.bincount(ytr, minlength=spec.num_classes).tolist(),
            "class_counts_test": np.bincount(yte, minlength=spec.num_classes).tolist(),
            "samples_per_class": SAMPLES_PER_CLASS,
            "pixel_mean": [round(float(v), 2) for v in sub.mean(axis=0)],
            "pixel_std": [round(float(v), 2) for v in sub.std(axis=0)],
            "class_names_verified": verified,
            "prepared_at": now_iso(),
        }
        write_json_atomic(info_path(spec.id), info)
        write_result(job_dir, outcome="succeeded", dataset=spec.id)
        print(f"prepared {spec.id}: train={len(xtr)} test={len(xte)} names_verified={verified}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        write_result(job_dir, outcome="failed", **error_fields("error", e))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    return run(Path(parser.parse_args().job_dir))


if __name__ == "__main__":
    sys.exit(main())
