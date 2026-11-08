"""UI 다국어 지원 (한국어 · English · 中文 · Tiếng Việt).

- 문구는 app/core/locales/{lang}.json 한 곳에만 둔다. web의 Jinja 템플릿과 브라우저 JS가 같은 사전을 쓴다.
- trainer·학습 프로세스는 언어를 모른다. 상태 파일에는 기본 언어(ko) 문장과 함께 메시지 키를 저장하고
  (`message` + `message_key` + `message_args`), 화면이 보는 사람의 언어로 다시 번역한다.
- HTTP 오류는 detail을 {"key", "args", "message"} 형태로 돌려준다.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

LANGS = {"ko": "한국어", "en": "English", "zh": "中文", "vi": "Tiếng Việt"}
HTML_LANG = {"ko": "ko", "en": "en", "zh": "zh-CN", "vi": "vi"}
# 숫자 묶음 표기. vi-VN 관례(10.000 / 0,5)는 화면의 소수점 지표(0.25, 39.4%)와 섞이면 헷갈리므로 en-US 형식을 쓴다.
NUMBER_LOCALE = {"ko": "ko-KR", "en": "en-US", "zh": "zh-CN", "vi": "en-US"}
DEFAULT_LANG = "ko"
COOKIE = "lang"
LOCALE_DIR = Path(__file__).resolve().parent / "locales"
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


@lru_cache(maxsize=None)
def _load(lang: str) -> dict[str, str]:
    with open(LOCALE_DIR / f"{lang}.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def messages(lang: str) -> dict[str, str]:
    """요청 언어 사전. 빠진 키는 en → ko 순서로 채운다 (키 누락 점검: python -m app.core.i18n)."""
    lang = normalize(lang)
    out = dict(_load(DEFAULT_LANG))
    if lang != DEFAULT_LANG:
        out.update(_load("en"))
        out.update(_load(lang))
    return out


def normalize(lang: str | None) -> str:
    if not lang:
        return DEFAULT_LANG
    code = lang.strip().lower().replace("_", "-").split("-")[0]
    return code if code in LANGS else DEFAULT_LANG


def pick_lang(cookie: str | None, accept_language: str | None) -> str:
    """쿠키에 고른 언어가 있으면 그것, 없으면 브라우저 Accept-Language에서 지원 언어를 고른다."""
    if cookie and normalize(cookie) == cookie:
        return cookie
    for part in (accept_language or "").split(","):
        code = part.split(";")[0].strip().lower().split("-")[0]
        if code in LANGS:
            return code
    return DEFAULT_LANG


def fill(template: str, args: dict | None) -> str:
    args = args or {}
    return _PLACEHOLDER.sub(lambda m: str(args[m.group(1)]) if m.group(1) in args else m.group(0), template)


def t(key: str, lang: str = DEFAULT_LANG, **args) -> str:
    return fill(messages(lang).get(key, key), args)


# --- 상태 파일·API 응답에 실을 메시지 ---------------------------------------------

def localized(field: str, key: str, **args) -> dict:
    """{field: 한국어 문장, field_key: key, field_args: args} — 화면이 키로 다시 번역한다."""
    return {field: t(key, DEFAULT_LANG, **args), f"{field}_key": key, f"{field}_args": args or None}


def raw(field: str, text: str | None) -> dict:
    """번역할 수 없는 원문(예외 메시지 등). 예전 키가 남지 않게 함께 지운다."""
    return {field: text, f"{field}_key": None, f"{field}_args": None}


class AppError(ValueError):
    """화면에 보일 수 있는 오류. str()은 기본 언어 문장."""

    def __init__(self, key: str, **args) -> None:
        self.key, self.args_ = key, args
        super().__init__(t(key, DEFAULT_LANG, **args))

    def detail(self) -> dict:
        return {"key": self.key, "args": self.args_, "message": str(self)}


def error_fields(field: str, e: Exception) -> dict:
    """예외 → 상태 파일 필드. AppError면 키를 남기고, 아니면 원문 그대로."""
    if isinstance(e, AppError):
        return localized(field, e.key, **e.args_)
    return raw(field, f"{type(e).__name__}: {e}")


def detail(key: str, **args) -> dict:
    """HTTPException detail."""
    return {"key": key, "args": args, "message": t(key, DEFAULT_LANG, **args)}


# --- 다국어 지원 이전에 만든 상태 파일 ------------------------------------------------
# 예전 Job·모델 파일은 한국어 문장만 있다. 알려진 문장이면 키를 붙여 다른 언어로도 보이게 한다.

_LEGACY_EXACT = {
    "대기 중": "msg.queued",
    "실행 중": "msg.running",
    "완료": "msg.succeeded",
    "취소됨": "msg.cancelled",
    "대기 중 취소됨": "msg.cancelled_while_queued",
    "취소 요청됨: 현재 단계를 마치는 중": "msg.cancel_requested",
    "trainer가 재시작되어 중단됨 (interrupted)": "msg.interrupted_restart",
    "trainer 재시작으로 대기열에서 제거됨": "msg.dequeued_restart",
    "trainer 재시작으로 중단됨": "msg.interrupted_restart",
    "trainer 종료로 취소됨 (shutdown)": "msg.cancelled_shutdown",
    "디바이스 설정": "phase.device",
    "데이터 로딩": "phase.loading_data",
    "학습 중": "phase.training",
    "모델 로딩": "phase.loading_model",
    "지표 계산·그림 저장": "phase.metrics",
    "샘플 이미지 생성": "phase.samples",
    "다운로드·로딩 (처음 한 번 약 161MB)": "phase.download",
}
_LEGACY_PATTERNS = [
    (re.compile(r"^예측 중 \(test ([\d,]+)장\)$"), "phase.predicting", ("n",)),
    (re.compile(r"^프로세스가 신호 (\d+)로 종료됨 \(메모리 부족 가능성\)$"), "msg.killed_signal", ("signal",)),
]


def upgrade_legacy(d: dict | None, field: str) -> dict | None:
    if not d or d.get(f"{field}_key") or not isinstance(d.get(field), str):
        return d
    text = d[field]
    if text in _LEGACY_EXACT:
        d[f"{field}_key"] = _LEGACY_EXACT[text]
        return d
    for pattern, key, names in _LEGACY_PATTERNS:
        m = pattern.match(text)
        if m:
            d[f"{field}_key"] = key
            d[f"{field}_args"] = dict(zip(names, m.groups()))
            break
    return d


# --- 사전 점검: python -m app.core.i18n -------------------------------------------------

def check() -> list[str]:
    """언어별 키·자리표시자가 같은지, 코드가 쓰는 키가 사전에 모두 있는지 확인한다."""
    problems = []
    base = _load(DEFAULT_LANG)
    for lang in LANGS:
        d = _load(lang)
        for k in sorted(set(base) - set(d)):
            problems.append(f"[{lang}] missing key: {k}")
        for k in sorted(set(d) - set(base)):
            problems.append(f"[{lang}] extra key: {k}")
        for k in set(base) & set(d):
            if set(_PLACEHOLDER.findall(base[k])) != set(_PLACEHOLDER.findall(d[k])):
                problems.append(f"[{lang}] placeholder mismatch: {k}")
    app_dir = Path(__file__).resolve().parents[1]
    used = set()
    pat = re.compile(r"""\bth?\(\s*["']([a-z_]+\.[a-z0-9_.]+)["']|["']((?:msg|phase|err|mcnemar)\.[a-z0-9_]+)["']""")
    for f in [*app_dir.rglob("*.py"), *app_dir.rglob("*.html"), *app_dir.rglob("*.js")]:
        for m in pat.finditer(f.read_text(encoding="utf-8")):
            used.add(m.group(1) or m.group(2))
    used |= {f"nav.{k}" for k in ("dashboard", "datasets", "train", "models", "evaluate", "test")}
    used |= {f"state.{s}" for s in ("queued", "running", "succeeded", "failed", "cancelled", "ready", "training")}
    used |= {f"job.{s}" for s in ("train", "evaluate", "prepare")}
    from app.core.datasets import DATASETS
    used |= {f"dataset.{d}.{f}" for d in DATASETS for f in ("name", "description")}
    used |= set(_LEGACY_EXACT.values()) | {key for _, key, _ in _LEGACY_PATTERNS}
    for k in sorted(used - set(base)):
        problems.append(f"used in code but not in {DEFAULT_LANG}.json: {k}")
    for k in sorted(set(base) - used):
        problems.append(f"unused key: {k}")
    return problems


if __name__ == "__main__":
    issues = check()
    print("\n".join(issues) if issues else f"OK: {len(_load(DEFAULT_LANG))} keys x {len(LANGS)} languages")
    raise SystemExit(1 if issues else 0)
