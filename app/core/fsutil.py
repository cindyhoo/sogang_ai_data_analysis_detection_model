"""파일 기반 상태 공유 도구: 원자적 JSON 쓰기, JSONL 추가, ID 검증."""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.i18n import AppError

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,150}$")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def check_id(value: str, what: str = "id") -> str:
    """경로 조작(../ 등)을 막기 위해 ID 형식을 검사한다."""
    if not isinstance(value, str) or not SAFE_ID.match(value) or ".." in value:
        raise AppError("err.invalid_id", what=what, value=repr(value)[:80])
    return value


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _json_default(o: Any):
    if hasattr(o, "item"):          # numpy 스칼라
        return o.item()
    if hasattr(o, "tolist"):        # numpy 배열
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json_atomic(path: Path, data: Any) -> None:
    """임시 파일에 쓴 뒤 rename해서 읽는 쪽이 반쯤 쓰인 파일을 보지 않게 한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return default


def append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue            # 쓰는 중인 마지막 줄
    except FileNotFoundError:
        pass
    return out


def tail_text(path: Path, n: int = 50, max_bytes: int = 64_000) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return []
    lines = ANSI.sub("", data).replace("\r", "\n").splitlines()
    return [ln for ln in lines if ln.strip()][-n:]
