"""공통 설정: storage 경로와 환경 변수.

web(컨테이너)과 trainer(macOS 네이티브)가 같은 storage 폴더를 보지만 절대경로가 다르다.
그래서 모든 JSON에는 storage 기준 상대경로나 ID만 저장한다 (제안서 §3.2).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR") or PROJECT_ROOT / "storage").resolve()

# Keras는 import 시점에 KERAS_HOME을 읽으므로 keras를 import하기 전에 기본값을 정한다.
os.environ.setdefault("KERAS_HOME", str(STORAGE_DIR / "keras"))
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
KERAS_HOME = Path(os.environ["KERAS_HOME"])

MODELS_DIR = STORAGE_DIR / "models"
JOBS_DIR = STORAGE_DIR / "jobs"
DATASETS_DIR = STORAGE_DIR / "datasets"
LOGS_DIR = STORAGE_DIR / "logs"
WEB_SETTINGS_FILE = STORAGE_DIR / "web_settings.json"

TRAINER_URL = os.environ.get("TRAINER_URL", "http://127.0.0.1:8001").rstrip("/")
TRAINER_TOKEN = os.environ.get("TRAINER_TOKEN", "")
TRAINER_DEVICE = os.environ.get("TRAINER_DEVICE", "auto")      # auto | gpu | cpu
TRAINER_ENV = os.environ.get("TRAINER_ENV", "unknown")         # conda | venv | docker
MODEL_CACHE_SIZE = int(os.environ.get("MODEL_CACHE_SIZE", "2"))
MIN_FREE_MEM_GB = float(os.environ.get("MIN_FREE_MEM_GB", "1.0"))
CANCEL_GRACE_SEC = float(os.environ.get("CANCEL_GRACE_SEC", "10"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def ensure_dirs() -> None:
    for d in (STORAGE_DIR, KERAS_HOME, MODELS_DIR, JOBS_DIR, DATASETS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
