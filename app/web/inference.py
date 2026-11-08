"""web 추론: 모델 LRU 캐시 + 예측 (제안서 §8.1).

- 캐시는 최대 MODEL_CACHE_SIZE개(기본 2). 모델별 Lock으로 동시 호출을 직렬화한다.
- TensorFlow는 첫 예측 때 import한다 (web 시작을 빠르게, 유휴 메모리를 적게).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np

from app.core import config, registry
from app.core.i18n import AppError
from app.core.stats import softmax


class ModelNotReady(AppError):
    def __init__(self, model_id: str) -> None:
        super().__init__("err.predict_needs_ready_id", id=model_id)


class ModelCache:
    def __init__(self, size: int) -> None:
        self.size = max(1, size)
        self._models: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._load_locks: dict[str, threading.Lock] = {}

    def _load(self, model_id: str) -> dict:
        meta = registry.read_meta(model_id)
        if not meta or meta.get("status") != "ready" or not registry.model_file(model_id).exists():
            raise ModelNotReady(model_id)
        import keras

        t0 = time.time()
        model = keras.saving.load_model(str(registry.model_file(model_id)), compile=False)
        h, w, c = meta["input_shape"]
        model(np.zeros((1, h, w, c), dtype=np.float32), training=False)     # warm-up
        return {"model": model, "meta": meta, "lock": threading.Lock(), "load_sec": round(time.time() - t0, 2),
                "mtime": registry.model_file(model_id).stat().st_mtime}

    def get(self, model_id: str) -> dict:
        with self._lock:
            entry = self._models.get(model_id)
            if entry:
                self._models.move_to_end(model_id)
                return entry
            load_lock = self._load_locks.setdefault(model_id, threading.Lock())
        with load_lock:
            with self._lock:
                if model_id in self._models:
                    return self._models[model_id]
            entry = self._load(model_id)
            with self._lock:
                self._models[model_id] = entry
                while len(self._models) > self.size:
                    self._models.popitem(last=False)
            return entry

    def evict(self, model_id: str) -> None:
        with self._lock:
            self._models.pop(model_id, None)

    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._models)


cache = ModelCache(config.MODEL_CACHE_SIZE)


def predict(model_id: str, x: np.ndarray, top_k: int = 5) -> dict:
    entry = cache.get(model_id)
    meta = entry["meta"]
    t0 = time.time()
    with entry["lock"]:
        logits = entry["model"](x, training=False)
    probs = softmax(np.asarray(logits, dtype=np.float64))[0]
    latency = (time.time() - t0) * 1000
    k = max(1, min(int(top_k), len(probs)))
    order = np.argsort(-probs)[:k]
    names = meta["class_names"]
    return {
        "model_id": model_id,
        "top": [{"index": int(i), "class": names[int(i)], "prob": float(probs[i])} for i in order],
        "latency_ms": round(latency, 1),
    }
