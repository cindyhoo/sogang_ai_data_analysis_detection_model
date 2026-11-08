"""추론 전처리 (제안서 §8.1). Rescaling(1/255)은 모델 안에 있으므로 여기서는 0~255 값을 그대로 넘긴다."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps


def preprocess_image(img: Image.Image, size: int = 32) -> np.ndarray:
    img = img.convert("RGB")                                         # 웹캠 프레임(JPEG) → RGB
    img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)  # 중앙 정사각 크롭 + 축소
    return np.asarray(img, dtype=np.float32)[None, ...]              # (1, 32, 32, 3), 값 0~255


def array_to_png_b64(arr: np.ndarray, scale: int = 4) -> str:
    """모델이 실제로 본 32×32 이미지를 확대(nearest)해서 base64 PNG로 돌려준다."""
    a = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(a)
    if scale > 1:
        img = img.resize((a.shape[1] * scale, a.shape[0] * scale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
