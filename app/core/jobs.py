"""Job 상태 파일 (제안서 §10.3).

storage/jobs/{job_id}/
  params.json    요청 파라미터            (trainer 상주 프로세스가 씀)
  status.json    상태(queued/running/...) (trainer 상주 프로세스만 씀)
  runtime.json   진행 단계·디바이스·model_id (학습 자식 프로세스만 씀)
  progress.jsonl batch/epoch 진행 기록     (학습 자식 프로세스만 씀)
  result.json    최종 결과                 (학습 자식 프로세스만 씀)
  log.txt        자식 프로세스 표준출력
  cancel         취소 요청 플래그 파일
파일마다 쓰는 쪽을 하나로 정해 두어 동시 쓰기 충돌을 피한다.
"""
from __future__ import annotations

import secrets
import threading
from pathlib import Path

from app.core import config, i18n
from app.core.fsutil import (
    check_id, now_iso, read_json, read_jsonl, tail_text, timestamp_id, write_json_atomic,
)

JOB_TYPES = ("prepare", "train", "evaluate")
ACTIVE_STATES = ("queued", "running")
FINAL_STATES = ("succeeded", "failed", "cancelled")
_STATUS_LOCK = threading.Lock()   # trainer 안의 API 스레드와 실행 스레드가 함께 갱신


def job_dir(job_id: str) -> Path:
    return config.JOBS_DIR / check_id(job_id, "job_id")


def new_job_id(job_type: str) -> str:
    return f"{timestamp_id()}-{job_type}-{secrets.token_hex(2)}"


def create_job(job_type: str, params: dict) -> dict:
    if job_type not in JOB_TYPES:
        raise i18n.AppError("err.unknown_job_type", type=job_type)
    job_id = new_job_id(job_type)
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=False)
    write_json_atomic(d / "params.json", params)
    status = {
        "id": job_id,
        "type": job_type,
        "state": "queued",
        **i18n.localized("message", "msg.queued"),
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "params": params,
    }
    write_json_atomic(d / "status.json", status)
    return status


def read_status(job_id: str) -> dict | None:
    return read_json(job_dir(job_id) / "status.json")


def write_status(job_id: str, status: dict) -> None:
    write_json_atomic(job_dir(job_id) / "status.json", status)


def update_status(job_id: str, **fields) -> dict:
    with _STATUS_LOCK:
        status = read_status(job_id) or {"id": job_id}
        status.update(fields)
        write_status(job_id, status)
        return status


# --- 자식 프로세스(학습·평가·준비) 쪽 ---------------------------------------------

def write_runtime(jd: Path, **fields) -> None:
    rt = read_json(jd / "runtime.json", {}) or {}
    rt.update(fields)
    rt["updated_at"] = now_iso()
    write_json_atomic(jd / "runtime.json", rt)


def write_result(jd: Path, **fields) -> None:
    fields["finished_at"] = now_iso()
    write_json_atomic(jd / "result.json", fields)


def cancel_requested(jd: Path) -> bool:
    return (jd / "cancel").exists()


def request_cancel(job_id: str) -> None:
    (job_dir(job_id) / "cancel").touch()


# --- 조회 ------------------------------------------------------------------------

def list_jobs(limit: int = 50, job_type: str | None = None, states: tuple | None = None) -> list[dict]:
    if not config.JOBS_DIR.exists():
        return []
    out = []
    for d in sorted(config.JOBS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or d.name.startswith("."):
            continue
        st = read_json(d / "status.json")
        if not st:
            continue
        if job_type and st.get("type") != job_type:
            continue
        if states and st.get("state") not in states:
            continue
        st["runtime"] = i18n.upgrade_legacy(read_json(d / "runtime.json", {}) or {}, "phase")
        out.append(i18n.upgrade_legacy(st, "message"))
        if len(out) >= limit:
            break
    return out


def job_detail(job_id: str, log_lines: int = 40) -> dict | None:
    d = job_dir(job_id)
    st = read_json(d / "status.json")
    if not st:
        return None
    progress = read_jsonl(d / "progress.jsonl")
    epochs = [p for p in progress if p.get("type") == "epoch"]
    batches = [p for p in progress if p.get("type") == "batch"]
    i18n.upgrade_legacy(st, "message")
    st["runtime"] = i18n.upgrade_legacy(read_json(d / "runtime.json", {}) or {}, "phase")
    st["result"] = read_json(d / "result.json")
    st["epochs"] = epochs
    st["last_batch"] = batches[-1] if batches else None
    st["log_tail"] = tail_text(d / "log.txt", log_lines)
    st["cancel_requested"] = cancel_requested(d)
    return st
