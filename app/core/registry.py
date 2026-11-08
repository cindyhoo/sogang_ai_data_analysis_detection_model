"""파일 기반 모델 레지스트리 (제안서 §6).

storage/models/{model_id}/
  model.keras, meta.json, history.csv, summary.txt, evaluations/{eval_id}/...
새 모델과 평가 결과는 trainer가 쓰고, 이름·메모·태그 수정과 삭제는 web이 한다.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from app.core import config
from app.core.fsutil import check_id, read_json, write_json_atomic

EDITABLE_FIELDS = ("name", "memo", "tags")


def model_dir(model_id: str) -> Path:
    return config.MODELS_DIR / check_id(model_id, "model_id")


def model_file(model_id: str) -> Path:
    return model_dir(model_id) / "model.keras"


def read_meta(model_id: str) -> dict | None:
    return read_json(model_dir(model_id) / "meta.json")


def write_meta(model_id: str, meta: dict) -> None:
    write_json_atomic(model_dir(model_id) / "meta.json", meta)


def width_tag(width: float) -> str:
    txt = f"{float(width):g}"
    return txt if "." in txt else txt + ".0"


def new_model_id(ts: str, arch: str, width: float, dataset_id: str) -> str:
    base = f"{ts}_{arch}-w{width_tag(width)}_{dataset_id}"
    model_id, n = base, 2
    while model_dir(model_id).exists():
        model_id = f"{base}-{n}"
        n += 1
    return model_id


# --- 기본 모델 설정 (web 전용) ------------------------------------------------------

def get_default_model_id() -> str | None:
    return (read_json(config.WEB_SETTINGS_FILE, {}) or {}).get("default_model")


def set_default_model_id(model_id: str | None) -> None:
    settings = read_json(config.WEB_SETTINGS_FILE, {}) or {}
    settings["default_model"] = model_id
    write_json_atomic(config.WEB_SETTINGS_FILE, settings)


# --- 평가 결과 ----------------------------------------------------------------------

def evaluations_dir(model_id: str) -> Path:
    return model_dir(model_id) / "evaluations"


def list_evaluations(model_id: str) -> list[dict]:
    d = evaluations_dir(model_id)
    if not d.exists():
        return []
    out = []
    for e in sorted(d.iterdir(), reverse=True):
        m = read_json(e / "metrics.json") if e.is_dir() else None
        if m:
            out.append(m)
    return out


def eval_summary(m: dict) -> dict:
    keys = ("eval_id", "dataset", "split", "n", "loss", "top1", "top1_ci", "top5", "macro_f1", "ece", "created_at", "device")
    return {k: m.get(k) for k in keys}


def read_evaluation(model_id: str, eval_id: str) -> dict | None:
    return read_json(evaluations_dir(model_id) / check_id(eval_id, "eval_id") / "metrics.json")


# --- 목록·상세 ----------------------------------------------------------------------

def _with_summary(meta: dict, default_id: str | None) -> dict:
    evals = list_evaluations(meta["id"])
    meta = dict(meta)
    meta["latest_eval"] = eval_summary(evals[0]) if evals else None
    meta["num_evaluations"] = len(evals)
    meta["is_default"] = meta["id"] == default_id
    meta["has_file"] = model_file(meta["id"]).exists()
    return meta


def list_models() -> list[dict]:
    if not config.MODELS_DIR.exists():
        return []
    default_id = get_default_model_id()
    out = []
    for d in config.MODELS_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        meta = read_json(d / "meta.json")
        if meta and meta.get("id") == d.name:
            out.append(_with_summary(meta, default_id))
    out.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return out


def get_model(model_id: str) -> dict | None:
    meta = read_meta(model_id)
    if not meta:
        return None
    return _with_summary(meta, get_default_model_id())


def read_history(model_id: str) -> dict:
    """CSVLogger가 남긴 history.csv → {열 이름: [값...]}"""
    path = model_dir(model_id) / "history.csv"
    cols: dict[str, list] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for k, v in row.items():
                    try:
                        val = float(v)
                    except (TypeError, ValueError):
                        val = None
                    cols.setdefault(k, []).append(val)
    except FileNotFoundError:
        pass
    if "epoch" in cols:
        cols["epoch"] = [int(e) + 1 for e in cols["epoch"] if e is not None]
    return cols


def read_summary(model_id: str) -> str:
    try:
        return (model_dir(model_id) / "summary.txt").read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def update_fields(model_id: str, **fields) -> dict:
    meta = read_meta(model_id)
    if not meta:
        raise FileNotFoundError(model_id)
    for k in EDITABLE_FIELDS:
        if fields.get(k) is not None:
            meta[k] = fields[k]
    write_meta(model_id, meta)
    return meta


def delete_model(model_id: str) -> None:
    shutil.rmtree(model_dir(model_id))
    if get_default_model_id() == model_id:
        set_default_model_id(None)
