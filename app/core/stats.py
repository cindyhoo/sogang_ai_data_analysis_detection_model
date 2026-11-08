"""평가용 통계 도구 (제안서 §7.2): 신뢰구간, McNemar 검정, Calibration(ECE)."""
from __future__ import annotations

import math

import numpy as np

from app.core.i18n import AppError, localized


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율(정확도)의 Wilson 95% 신뢰구간."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """같은 test set에서 두 모델의 정답 여부를 짝지어 비교한다.

    b: A만 맞힘, c: B만 맞힘. b+c가 작으면(25 미만) 정확 이항검정, 아니면 연속성 보정 카이제곱.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    if a.shape != b_.shape:
        raise AppError("err.mcnemar_size")
    both = int(np.sum(a & b_))
    only_a = int(np.sum(a & ~b_))
    only_b = int(np.sum(~a & b_))
    neither = int(np.sum(~a & ~b_))
    n_disc = only_a + only_b
    if n_disc == 0:
        stat, p, method = 0.0, 1.0, "mcnemar.none"
    elif n_disc < 25:
        k = min(only_a, only_b)
        p = min(1.0, 2 * sum(math.comb(n_disc, i) for i in range(k + 1)) / 2 ** n_disc)
        stat, method = float(k), "mcnemar.exact"
    else:
        stat = (abs(only_a - only_b) - 1) ** 2 / n_disc
        p = math.erfc(math.sqrt(stat / 2))          # 자유도 1 카이제곱의 상측 확률
        method = "mcnemar.chi2"
    return {
        "n": int(a.size),
        "both_correct": both,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_wrong": neither,
        "statistic": float(stat),
        "p_value": float(p),
        **localized("method", method),
        "significant_at_0_05": bool(p < 0.05),
    }


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    """확신도 구간별 (평균 확신도, 실제 정확도)와 ECE."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, n = [], 0.0, len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt:
            acc = float(correct[mask].mean())
            avg_conf = float(conf[mask].mean())
            ece += cnt / n * abs(acc - avg_conf)
        else:
            acc = avg_conf = None
        bins.append({"lo": float(lo), "hi": float(hi), "count": cnt, "accuracy": acc, "confidence": avg_conf})
    return float(ece), bins
