"""web 서비스 (제안서 §3.2, §9, §10.1): UI 페이지 + REST API + 웹캠 추론.

Docker 컨테이너(linux/arm64)에서 실행되고, 학습·평가·데이터셋 준비는 host의 trainer에 요청한다.
Job·모델 상태는 공유 storage 폴더의 파일을 직접 읽는다.
"""
from __future__ import annotations

import io
from pathlib import Path

import httpx
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.core import config, i18n, jobs, registry
from app.core.datasets import DATASETS, dataset_status
from app.core.fsutil import check_id, read_json
from app.core.preprocess import array_to_png_b64, preprocess_image
from app.core.runtime import VERSION_KEYS, env_info
from app.core.stats import mcnemar
from app.web import inference

WEB_DIR = Path(__file__).resolve().parent
WEB_ENV = "docker" if Path("/.dockerenv").exists() else "native"

config.ensure_dirs()
app = FastAPI(title="CIFAR-100 CNN PoC")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


def asset(path: str) -> str:
    """정적 파일 URL에 수정 시각을 붙여, 코드가 바뀌면 브라우저 캐시를 우회한다."""
    f = WEB_DIR / "static" / path
    v = int(f.stat().st_mtime) if f.exists() else 0
    return f"/static/{path}?v={v}"


templates.env.globals["asset"] = asset
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
app.mount("/files/datasets", StaticFiles(directory=config.DATASETS_DIR), name="dataset-files")
app.mount("/files/models", StaticFiles(directory=config.MODELS_DIR), name="model-files")


# --- trainer 연동 -------------------------------------------------------------------

def trainer_request(method: str, path: str, json: dict | None = None, timeout: float = 5.0) -> dict:
    headers = {"X-Trainer-Token": config.TRAINER_TOKEN} if config.TRAINER_TOKEN else {}
    try:
        r = httpx.request(method, config.TRAINER_URL + path, json=json, timeout=timeout, headers=headers)
    except httpx.HTTPError:
        raise HTTPException(503, i18n.detail("err.trainer_unreachable", url=config.TRAINER_URL)) from None
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise HTTPException(r.status_code, detail)
    return r.json()


def _valid(value: str, what: str) -> str:
    try:
        return check_id(value, what)
    except i18n.AppError as e:
        raise HTTPException(400, e.detail()) from None


def _model_or_404(model_id: str) -> dict:
    meta = registry.get_model(_valid(model_id, "model_id"))
    if not meta:
        raise HTTPException(404, i18n.detail("err.model_not_found"))
    return i18n.upgrade_legacy(meta, "error")


# --- 페이지 -------------------------------------------------------------------------

def request_lang(request: Request) -> str:
    return i18n.pick_lang(request.cookies.get(i18n.COOKIE), request.headers.get("accept-language"))


def page(request: Request, name: str, active: str, **ctx) -> HTMLResponse:
    lang = request_lang(request)
    return templates.TemplateResponse(request, name, {
        "active": active,
        "lang": lang,
        "langs": i18n.LANGS,
        "html_lang": i18n.HTML_LANG[lang],
        "t": lambda key, **kw: i18n.t(key, lang, **kw),
        "i18n_boot": {"lang": lang, "locale": i18n.NUMBER_LOCALE[lang], "cookie": i18n.COOKIE,
                      "messages": i18n.messages(lang)},
        **ctx,
    })


@app.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return page(request, "dashboard.html", "dashboard")


@app.get("/datasets", response_class=HTMLResponse)
def datasets_page(request: Request):
    return page(request, "datasets.html", "datasets")


@app.get("/train", response_class=HTMLResponse)
def train_page(request: Request):
    return page(request, "train.html", "train")


@app.get("/models", response_class=HTMLResponse)
def models_page(request: Request):
    return page(request, "models.html", "models")


@app.get("/models/{model_id}", response_class=HTMLResponse)
def model_detail_page(request: Request, model_id: str):
    _model_or_404(model_id)
    return page(request, "model_detail.html", "models", model_id=model_id)


@app.get("/evaluate", response_class=HTMLResponse)
def evaluate_page(request: Request):
    return page(request, "evaluate.html", "evaluate")


@app.get("/test", response_class=HTMLResponse)
def test_page(request: Request):
    return page(request, "test.html", "test")


# --- 상태 ---------------------------------------------------------------------------

@app.get("/api/health")
def api_health():
    web = env_info(WEB_ENV)
    try:
        trainer = trainer_request("GET", "/health", timeout=2.0)
    except HTTPException as e:
        trainer = {"status": "down", "error": e.detail}
    mismatch = []
    if trainer.get("versions"):
        mismatch = [{"package": k, "web": web.get(k), "trainer": trainer["versions"].get(k)}
                    for k in VERSION_KEYS if web.get(k) != trainer["versions"].get(k)]
    active = jobs.list_jobs(limit=20, states=jobs.ACTIVE_STATES)
    return {
        "status": "ok",
        "web": {"versions": web, "loaded_models": inference.cache.loaded(), "trainer_url": config.TRAINER_URL},
        "trainer": trainer,
        "version_mismatch": mismatch,
        "active_jobs": [{k: j.get(k) for k in ("id", "type", "state", "message", "message_key", "message_args")}
                        for j in active],
        "training_active": any(j["type"] == "train" and j["state"] == "running" for j in active),
    }


# --- 데이터셋 -----------------------------------------------------------------------

@app.get("/api/datasets")
def api_datasets():
    out = []
    for ds_id in DATASETS:
        d = dataset_status(ds_id)
        d.pop("class_counts_train", None)
        d.pop("class_counts_test", None)
        d["active_job"] = next((j["id"] for j in jobs.list_jobs(limit=50, job_type="prepare",
                                                                 states=jobs.ACTIVE_STATES)
                                if j.get("params", {}).get("dataset") == ds_id), None)
        out.append(d)
    return out


@app.get("/api/datasets/{dataset_id}")
def api_dataset(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(404, i18n.detail("err.dataset_not_found"))
    return dataset_status(dataset_id)


@app.post("/api/datasets/{dataset_id}/prepare")
def api_prepare(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(404, i18n.detail("err.dataset_not_found"))
    return trainer_request("POST", "/jobs", {"type": "prepare", "params": {"dataset": dataset_id}})


# --- 학습·Job -----------------------------------------------------------------------

@app.post("/api/train")
def api_train(params: dict = Body(...)):
    return trainer_request("POST", "/jobs", {"type": "train", "params": params})


@app.get("/api/jobs")
def api_jobs(type: str | None = None, state: str | None = None, limit: int = 30):
    states = tuple(state.split(",")) if state else None
    return jobs.list_jobs(limit=min(limit, 200), job_type=type, states=states)


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    d = jobs.job_detail(_valid(job_id, "job_id"))
    if not d:
        raise HTTPException(404, i18n.detail("err.job_not_found"))
    return d


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str):
    return trainer_request("POST", f"/jobs/{_valid(job_id, 'job_id')}/cancel")


# --- 모델 관리 ----------------------------------------------------------------------

class ModelPatch(BaseModel):
    name: str | None = Field(None, max_length=80)
    memo: str | None = Field(None, max_length=2000)
    tags: list[str] | None = None
    is_default: bool | None = None


@app.get("/api/models")
def api_models(status: str | None = None):
    models = registry.list_models()
    if status:
        models = [m for m in models if m.get("status") == status]
    for m in models:
        m.pop("class_names", None)
        i18n.upgrade_legacy(m, "error")
    return models


@app.get("/api/models/{model_id}")
def api_model(model_id: str):
    meta = _model_or_404(model_id)
    history = registry.read_history(model_id)
    job = jobs.job_detail(meta["job_id"], log_lines=20) if meta.get("job_id") else None
    if not history.get("epoch") and job:
        eps = job.get("epochs") or []
        history = {k: [e.get(k) for e in eps] for k in
                   ("epoch", "loss", "acc", "top5", "val_loss", "val_acc", "val_top5")}
    return {
        "meta": meta,
        "history": history,
        "summary": registry.read_summary(model_id),
        "evaluations": [registry.eval_summary(e) for e in registry.list_evaluations(model_id)],
        "job": job,
    }


@app.patch("/api/models/{model_id}")
def api_model_patch(model_id: str, patch: ModelPatch):
    meta = _model_or_404(model_id)
    if meta.get("status") == "training":
        raise HTTPException(409, i18n.detail("err.model_training_edit"))
    tags = None
    if patch.tags is not None:
        tags = [t.strip()[:30] for t in patch.tags if t and t.strip()][:10]
    registry.update_fields(model_id, name=patch.name.strip() if patch.name else None,
                           memo=patch.memo, tags=tags)
    if patch.is_default is True:
        if meta.get("status") != "ready":
            raise HTTPException(409, i18n.detail("err.default_needs_ready"))
        registry.set_default_model_id(model_id)
    elif patch.is_default is False and registry.get_default_model_id() == model_id:
        registry.set_default_model_id(None)
    return registry.get_model(model_id)


@app.delete("/api/models/{model_id}")
def api_model_delete(model_id: str):
    meta = _model_or_404(model_id)
    if meta.get("status") == "training":
        raise HTTPException(409, i18n.detail("err.model_training_delete"))
    for j in jobs.list_jobs(limit=50, job_type="evaluate", states=jobs.ACTIVE_STATES):
        if j.get("params", {}).get("model_id") == model_id:
            raise HTTPException(409, i18n.detail("err.model_eval_running"))
    inference.cache.evict(model_id)
    registry.delete_model(model_id)
    return {"deleted": model_id}


@app.get("/api/models/{model_id}/download")
def api_model_download(model_id: str):
    _model_or_404(model_id)
    path = registry.model_file(model_id)
    if not path.exists():
        raise HTTPException(404, i18n.detail("err.model_file_missing"))
    return FileResponse(path, filename=f"{model_id}.keras", media_type="application/octet-stream")


# --- 평가 ---------------------------------------------------------------------------

@app.post("/api/models/{model_id}/evaluate")
def api_evaluate(model_id: str, body: dict | None = Body(default=None)):
    meta = _model_or_404(model_id)
    if meta.get("status") != "ready":
        raise HTTPException(409, i18n.detail("err.eval_needs_ready"))
    dataset = (body or {}).get("dataset") or meta["dataset"]
    return trainer_request("POST", "/jobs", {"type": "evaluate", "params": {"model_id": model_id, "dataset": dataset}})


@app.get("/api/models/{model_id}/evaluations")
def api_evaluations(model_id: str):
    _model_or_404(model_id)
    return [registry.eval_summary(e) for e in registry.list_evaluations(model_id)]


@app.get("/api/models/{model_id}/evaluations/{eval_id}")
def api_evaluation(model_id: str, eval_id: str):
    _model_or_404(model_id)
    m = registry.read_evaluation(model_id, _valid(eval_id, "eval_id"))
    if not m:
        raise HTTPException(404, i18n.detail("err.eval_not_found"))
    m["base_url"] = f"/files/models/{model_id}/evaluations/{eval_id}/"
    return m


@app.get("/api/compare/mcnemar")
def api_mcnemar(a_model: str, a_eval: str, b_model: str, b_eval: str):
    """같은 test set에서 두 모델의 정답 여부를 McNemar 검정으로 비교 (제안서 §7.2 P2)."""
    loaded = []
    for mid, eid in ((a_model, a_eval), (b_model, b_eval)):
        _model_or_404(mid)
        m = registry.read_evaluation(mid, _valid(eid, "eval_id"))
        path = registry.evaluations_dir(mid) / eid / "preds.npz"
        if not m or not path.exists():
            raise HTTPException(404, i18n.detail("err.eval_not_found_id", id=f"{mid}/{eid}"))
        with np.load(path) as z:
            loaded.append((m, z["y_true"].copy(), z["y_pred"].copy()))
    (ma, ya, pa), (mb, yb, pb) = loaded
    if ma["dataset"] != mb["dataset"] or ya.shape != yb.shape or not np.array_equal(ya, yb):
        raise HTTPException(400, i18n.detail("err.mcnemar_mismatch"))
    res = mcnemar(pa == ya, pb == yb)
    res.update({"a": {"model_id": a_model, "eval_id": a_eval, "top1": ma["top1"]},
                "b": {"model_id": b_model, "eval_id": b_eval, "top1": mb["top1"]},
                "dataset": ma["dataset"]})
    return res


# --- 웹캠 추론 ----------------------------------------------------------------------

@app.post("/api/predict")
async def api_predict(model_id: str = Form(...), file: UploadFile = File(...), top_k: int = Form(5)):
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, i18n.detail("err.frame_too_large"))
    meta = registry.read_meta(_valid(model_id, "model_id"))
    if not meta or meta.get("status") != "ready":
        raise HTTPException(400, i18n.detail("err.predict_needs_ready"))
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, i18n.detail("err.bad_image")) from None
    x = preprocess_image(img, int(meta["input_shape"][0]))
    try:
        result = await run_in_threadpool(inference.predict, model_id, x, top_k)
    except i18n.AppError as e:
        raise HTTPException(400, e.detail()) from None
    result["input_png"] = array_to_png_b64(x[0], scale=4)
    return result


@app.get("/api/settings")
def api_settings():
    return read_json(config.WEB_SETTINGS_FILE, {}) or {}
