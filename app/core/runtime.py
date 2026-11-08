"""실행 환경 정보와 학습 디바이스 선택 (제안서 §5.3)."""
from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

from app.core.i18n import AppError

VERSION_KEYS = ("tensorflow", "keras", "numpy")   # web·trainer에서 반드시 같아야 하는 패키지


def _v(pkg: str):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def env_info(env_label: str) -> dict:
    """TensorFlow를 import하지 않고 버전만 읽는다 (상주 프로세스 메모리 절약)."""
    return {
        "env": env_label,
        "platform": f"{platform.system()}-{platform.machine()}",
        "python": platform.python_version(),
        "tensorflow": _v("tensorflow"),
        "tensorflow_metal": _v("tensorflow-metal"),
        "keras": _v("keras"),
        "numpy": _v("numpy"),
    }


def setup_device(pref: str) -> str:
    """pref: auto | gpu | cpu. TensorFlow가 GPU를 초기화하기 전에 호출해야 한다."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")   # tensorflow-metal이 정상이면 GPU 1개가 보임
    if pref == "gpu" and not gpus:
        raise AppError("err.no_gpu")
    if pref == "cpu" or not gpus:
        if gpus:
            tf.config.set_visible_devices([], "GPU")
        return "CPU"
    return "GPU (Metal)" if platform.system() == "Darwin" else "GPU"
