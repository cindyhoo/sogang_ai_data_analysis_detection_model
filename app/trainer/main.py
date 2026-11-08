"""trainer 서비스 (제안서 §3.2, §10.2, §10.3).

- macOS 네이티브(conda 환경)에서 127.0.0.1:8001로 실행되어 Metal GPU로 학습한다.
- 상주 프로세스는 TensorFlow를 import하지 않는다. Job마다 자식 프로세스를 띄우고, 한 번에 1개만 실행한다.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import psutil
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.core import config, i18n, jobs, registry
from app.core.datasets import DATASETS
from app.core.fsutil import now_iso, read_json
from app.core.runtime import env_info

L = i18n.localized

PROBE_CODE = (
    "import json, tensorflow as tf;"
    "g = tf.config.list_physical_devices('GPU');"
    "print('PROBE ' + json.dumps({'gpus': len(g)}))"
)


# --- 요청 파라미터 검증 -------------------------------------------------------------

class TrainParams(BaseModel):
    name: str = Field("", max_length=80)
    dataset: str = "cifar100_fine"
    arch: Literal["xception_mini"] = "xception_mini"
    width: float = Field(1.0, ge=0.25, le=1.0)
    epochs: int = Field(25, ge=1, le=100)
    batch_size: Literal[32, 64, 128, 256] = 128
    lr: float = Field(3e-4, ge=1e-5, le=1e-2)
    dropout: float = Field(0.25, ge=0.0, le=0.6)
    augment: list[Literal["flip", "rotation", "translation"]] = ["flip", "rotation"]
    val_split: float = Field(0.1, ge=0.05, le=0.2)
    train_subset: float = Field(1.0, ge=0.1, le=1.0)
    patience: int = Field(5, ge=0, le=20)
    seed: int = Field(1337, ge=0, le=2**31 - 1)
    device: Literal["auto", "gpu", "cpu"] = "auto"

    @field_validator("dataset")
    @classmethod
    def _dataset(cls, v):
        if v not in DATASETS:
            raise ValueError(f"unknown dataset: {v}")
        return v


class EvaluateParams(BaseModel):
    model_id: str
    dataset: str | None = None
    device: Literal["auto", "gpu", "cpu"] = "auto"


class PrepareParams(BaseModel):
    dataset: str

    @field_validator("dataset")
    @classmethod
    def _dataset(cls, v):
        if v not in DATASETS:
            raise ValueError(f"unknown dataset: {v}")
        return v


PARAM_MODELS = {"train": TrainParams, "evaluate": EvaluateParams, "prepare": PrepareParams}


class JobRequest(BaseModel):
    type: Literal["prepare", "train", "evaluate"]
    params: dict = {}


# --- Job 실행기 ---------------------------------------------------------------------

class JobRunner:
    def __init__(self) -> None:
        self.q: queue.Queue[str] = queue.Queue()
        self.current: str | None = None
        self.proc: subprocess.Popen | None = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="job-runner", daemon=True)
        self.probe: dict = {"state": "pending"}

    # 시작 시 정리: 이전 실행에서 남은 Job·모델 상태를 바로잡는다
    def recover(self) -> None:
        for st in jobs.list_jobs(limit=10_000, states=jobs.ACTIVE_STATES):
            if st["state"] == "running":
                jobs.update_status(st["id"], state="failed", **L("message", "msg.interrupted_restart"),
                                   finished_at=now_iso())
            else:
                jobs.update_status(st["id"], state="cancelled", **L("message", "msg.dequeued_restart"),
                                   finished_at=now_iso())
        for meta in registry.list_models():
            if meta.get("status") == "training":
                m = registry.read_meta(meta["id"])
                m.update(status="failed", **L("error", "msg.interrupted_restart"), finished_at=now_iso())
                registry.write_meta(meta["id"], m)

    def start(self) -> None:
        self.thread.start()
        threading.Thread(target=self._probe_device, name="device-probe", daemon=True).start()

    def _probe_device(self) -> None:
        """TensorFlow를 별도 프로세스에서 한 번 import해 GPU가 보이는지 확인한다."""
        self.probe = {"state": "probing"}
        try:
            out = subprocess.run([sys.executable, "-c", PROBE_CODE], capture_output=True, text=True,
                                 timeout=180, cwd=config.PROJECT_ROOT, env=os.environ.copy())
            line = next((ln for ln in out.stdout.splitlines() if ln.startswith("PROBE ")), None)
            if not line:
                tail = (out.stderr or out.stdout or "").strip().splitlines()
                raise RuntimeError(tail[-1] if tail else "TensorFlow probe failed")
            gpus = json.loads(line[6:])["gpus"]
            self.probe = {"state": "done", "gpu_available": gpus > 0, "gpus": gpus, "checked_at": now_iso()}
        except Exception as e:  # noqa: BLE001
            self.probe = {"state": "error", "gpu_available": False, "error": str(e), "checked_at": now_iso()}

    def effective_device(self) -> str:
        pref = config.TRAINER_DEVICE
        if pref == "cpu":
            return "CPU"
        if self.probe.get("state") != "done":
            return "probing" if self.probe.get("state") in ("pending", "probing") else "CPU"
        if self.probe.get("gpu_available"):
            return "GPU (Metal)" if sys.platform == "darwin" else "GPU"
        return "CPU"

    def submit(self, job_type: str, params: dict) -> dict:
        status = jobs.create_job(job_type, params)
        self.q.put(status["id"])
        return status

    def cancel(self, job_id: str) -> dict:
        st = jobs.read_status(job_id)
        if not st:
            raise KeyError(job_id)
        if st["state"] in jobs.FINAL_STATES:
            return st
        jobs.request_cancel(job_id)
        if st["state"] == "queued":
            return jobs.update_status(job_id, state="cancelled", **L("message", "msg.cancelled_while_queued"),
                                      finished_at=now_iso())
        return jobs.update_status(job_id, **L("message", "msg.cancel_requested"))

    def queue_size(self) -> int:
        return self.q.qsize()

    def _loop(self) -> None:
        while not self.stopping.is_set():
            try:
                job_id = self.q.get(timeout=1)
            except queue.Empty:
                continue
            st = jobs.read_status(job_id)
            if not st or st["state"] != "queued":
                continue
            if not self._wait_for_memory(job_id):
                continue
            self.current = job_id
            try:
                self._run(job_id, st["type"])
            except Exception as e:  # noqa: BLE001
                jobs.update_status(job_id, state="failed", **L("message", "msg.run_error", error=str(e)),
                                   finished_at=now_iso())
            finally:
                self.current = None
                self.proc = None

    def _wait_for_memory(self, job_id: str) -> bool:
        """가용 메모리가 기준보다 적으면 Job을 대기시킨다 (8GB RAM 대응)."""
        while not self.stopping.is_set():
            jd = jobs.job_dir(job_id)
            if jobs.cancel_requested(jd):
                return False
            avail = psutil.virtual_memory().available / 1024 ** 3
            if avail >= config.MIN_FREE_MEM_GB:
                return True
            jobs.update_status(job_id, **L("message", "msg.low_memory", avail=f"{avail:.1f}",
                                           need=config.MIN_FREE_MEM_GB))
            time.sleep(5)
        return False

    def _run(self, job_id: str, job_type: str) -> None:
        jd = jobs.job_dir(job_id)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with open(jd / "log.txt", "ab") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", f"app.core.{job_type}", "--job-dir", str(jd)],
                cwd=config.PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            self.proc = proc
            jobs.update_status(job_id, state="running", **L("message", "msg.running"), started_at=now_iso(),
                               pid=proc.pid)
            cancel_at = term_at = None
            while proc.poll() is None:
                if cancel_at is None and jobs.cancel_requested(jd):
                    cancel_at = time.time()
                if cancel_at and term_at is None and time.time() - cancel_at > config.CANCEL_GRACE_SEC:
                    proc.terminate()                        # 유예시간 안에 멈추지 않으면 SIGTERM
                    term_at = time.time()
                if term_at and time.time() - term_at > 10:
                    proc.kill()
                time.sleep(0.5)
        rc = proc.returncode
        result = read_json(jd / "result.json", {}) or {}
        runtime = read_json(jd / "runtime.json", {}) or {}
        outcome = result.get("outcome")
        if not outcome:
            outcome = "cancelled" if jobs.cancel_requested(jd) else ("succeeded" if rc == 0 else "failed")
        if rc != 0 and outcome == "succeeded":
            outcome = "failed"
        if outcome == "failed":
            if result.get("error_key"):
                msg = L("message", result["error_key"], **(result.get("error_args") or {}))
            elif result.get("error"):
                msg = i18n.raw("message", result["error"])
            elif rc is not None and rc < 0:
                msg = L("message", "msg.killed_signal", signal=-rc)
            else:
                msg = L("message", "msg.exit_code", code=rc)
        else:
            msg = L("message", {"succeeded": "msg.succeeded", "cancelled": "msg.cancelled"}[outcome])
        model_id = result.get("model_id") or runtime.get("model_id")
        jobs.update_status(job_id, state=outcome, **msg, finished_at=now_iso(), exit_code=rc,
                           model_id=model_id, eval_id=result.get("eval_id") or runtime.get("eval_id"),
                           device=runtime.get("device"))
        # 자식 프로세스가 강제 종료되어 모델 상태를 마무리하지 못한 경우 정리
        if job_type == "train" and model_id:
            meta = registry.read_meta(model_id)
            if meta and meta.get("status") == "training":
                err = i18n.raw("error", None) if outcome == "cancelled" else {
                    "error": msg["message"], "error_key": msg["message_key"], "error_args": msg["message_args"]}
                meta.update(status="cancelled" if outcome == "cancelled" else "failed", **err, finished_at=now_iso())
                registry.write_meta(model_id, meta)

    def shutdown(self) -> None:
        """trainer 종료 시 실행 중인 Job 프로세스도 정리해 고아 프로세스를 남기지 않는다."""
        self.stopping.set()
        proc, job_id = self.proc, self.current
        if proc and proc.poll() is None and job_id:
            jobs.request_cancel(job_id)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            jobs.update_status(job_id, state="cancelled", **L("message", "msg.cancelled_shutdown"),
                               finished_at=now_iso(), exit_code=proc.returncode)
            model_id = (read_json(jobs.job_dir(job_id) / "runtime.json", {}) or {}).get("model_id")
            if model_id:
                meta = registry.read_meta(model_id)
                if meta and meta.get("status") == "training":
                    meta.update(status="cancelled", finished_at=now_iso())
                    registry.write_meta(model_id, meta)


runner = JobRunner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    runner.recover()
    runner.start()
    yield
    runner.shutdown()


app = FastAPI(title="CNN PoC trainer", lifespan=lifespan)


def check_token(x_trainer_token: str | None = Header(default=None)) -> None:
    if config.TRAINER_TOKEN and x_trainer_token != config.TRAINER_TOKEN:
        raise HTTPException(401, i18n.detail("err.bad_token"))


@app.get("/health")
def health():
    vm = psutil.virtual_memory()
    return {
        "status": "ok",
        "service": "trainer",
        "device": runner.effective_device(),
        "device_pref": config.TRAINER_DEVICE,
        "probe": runner.probe,
        "versions": env_info(config.TRAINER_ENV),
        "memory": {"available_gb": round(vm.available / 1024 ** 3, 2), "total_gb": round(vm.total / 1024 ** 3, 2),
                   "min_free_gb": config.MIN_FREE_MEM_GB},
        "current_job": runner.current,
        "queue_size": runner.queue_size(),
        "time": now_iso(),
    }


@app.post("/jobs", dependencies=[Depends(check_token)])
def create_job(req: JobRequest):
    try:
        params = PARAM_MODELS[req.type](**req.params).model_dump()
    except Exception as e:  # pydantic ValidationError 등
        raise HTTPException(422, i18n.detail("err.invalid_params", detail=str(e))) from None
    if req.type == "evaluate":
        meta = registry.read_meta(params["model_id"]) if _safe_id(params["model_id"]) else None
        if not meta or meta.get("status") != "ready":
            raise HTTPException(400, i18n.detail("err.eval_needs_ready"))
    return runner.submit(req.type, params)


def _safe_id(v: str) -> bool:
    try:
        registry.model_dir(v)
        return True
    except ValueError:
        return False


@app.get("/jobs")
def list_jobs(limit: int = 50, type: str | None = None):
    return jobs.list_jobs(limit=min(limit, 500), job_type=type)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        d = jobs.job_detail(job_id)
    except ValueError:
        d = None
    if not d:
        raise HTTPException(404, i18n.detail("err.job_not_found"))
    return d


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(check_token)])
def cancel_job(job_id: str):
    try:
        return runner.cancel(job_id)
    except (KeyError, ValueError):
        raise HTTPException(404, i18n.detail("err.job_not_found")) from None
