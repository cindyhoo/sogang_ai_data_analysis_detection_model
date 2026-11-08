#!/usr/bin/env bash
# CIFAR-100 CNN PoC 실행 스크립트 (제안서 §11.6)
#
# 사용법: ./run.sh [up|down|status|logs [trainer|web]|setup|reset]
#   TRAINER_MODE=native(기본)|docker    trainer 실행 위치
#   TRAINER_ENV=auto(기본)|conda|venv    네이티브 trainer의 Python 환경
#   TRAINER_DEVICE=auto(기본)|gpu|cpu    학습 디바이스
#   WEB_MODE=docker(기본)|native         web 실행 위치 (native: Docker 없이 점검·개발할 때)
#   TRAINER_HOST=127.0.0.1(기본)         컨테이너에서 연결이 안 되면 0.0.0.0 + TRAINER_TOKEN 사용 (§13)
#   NO_OPEN=1                           up 후 브라우저를 자동으로 열지 않음
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"                         # 경로에 공백이 있으므로 변수는 항상 "따옴표"로 감싼다

TRAINER_MODE="${TRAINER_MODE:-native}"
TRAINER_ENV="${TRAINER_ENV:-auto}"
WEB_MODE="${WEB_MODE:-docker}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cnn-poc-trainer}"
TRAINER_HOST="${TRAINER_HOST:-127.0.0.1}"
export TRAINER_DEVICE="${TRAINER_DEVICE:-auto}"
export TRAINER_TOKEN="${TRAINER_TOKEN:-}"
WEB_URL="http://localhost:8000"
TRAINER_HEALTH="http://127.0.0.1:8001/health"
VENV="$ROOT/.venv-trainer"
RUN_DIR="$ROOT/.run"
LOG_DIR="$ROOT/storage/logs"
PY=""                                 # 네이티브 Python 경로 (setup_trainer가 채움)

# Docker Desktop을 설치만 하고 처음 실행하기 전에는 docker CLI가 PATH에 없을 수 있다.
if ! command -v docker >/dev/null && [ -x /Applications/Docker.app/Contents/Resources/bin/docker ]; then
  export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
fi

die() { echo "❌ $*"; exit 1; }

wait_for() {                          # wait_for <url> <이름>
  for _ in $(seq 1 60); do
    curl -sf "$1" >/dev/null && return 0
    sleep 2
  done
  die "$2이(가) 120초 안에 응답하지 않습니다. ./run.sh logs $2 로 확인하세요."
}

check_docker() {
  command -v docker >/dev/null || die "Docker가 없습니다. Docker Desktop을 설치하세요. (Docker 없이 점검: WEB_MODE=native ./run.sh up)"
  docker info >/dev/null 2>&1 || die "Docker Desktop을 먼저 실행하세요. (Docker 없이 점검: WEB_MODE=native ./run.sh up)"
  local need=1.8 mem
  if [ "$TRAINER_MODE" = docker ]; then need=3.5; fi
  mem=$(docker info --format '{{.MemTotal}}' | awk '{printf "%.1f", $1/1024/1024/1024}')
  if awk -v m="$mem" -v n="$need" 'BEGIN{exit !(m < n)}'; then
    echo "⚠️  Docker 메모리 ${mem}GB: ${TRAINER_MODE} 모드는 ${need}GB 이상을 권장합니다 (Settings → Resources)."
  fi
}

docker_ready() { command -v docker >/dev/null && docker info >/dev/null 2>&1; }

setup_conda() {                       # 인자 update: 기존 환경을 의존성 파일 기준으로 갱신
  command -v conda >/dev/null || die "conda가 없습니다: brew install --cask miniforge (또는 TRAINER_ENV=venv)"
  conda info | grep -E "platform : osx-arm64" >/dev/null \
    || die "x86(Rosetta)용 conda입니다. arm64용 Miniforge를 설치하세요."
  if ! conda env list | grep -E "^${CONDA_ENV_NAME}[[:space:]]" >/dev/null; then
    echo "📦 conda 환경 ${CONDA_ENV_NAME} 생성 중 (처음 한 번, 수 분 소요)..."
    conda env create -n "$CONDA_ENV_NAME" -f environment-trainer.yml
  elif [ "${1:-}" = update ]; then
    conda env update -n "$CONDA_ENV_NAME" -f environment-trainer.yml --prune
  fi
  PY="$(conda run -n "$CONDA_ENV_NAME" python -c 'import sys; print(sys.executable)')"
}

setup_venv() {
  if [ ! -x "$VENV/bin/python" ]; then
    command -v python3.11 >/dev/null || die "python3.11이 없습니다: brew install python@3.11"
    python3.11 -m venv "$VENV"
    "$VENV/bin/pip" install -U pip
    "$VENV/bin/pip" install -r requirements-trainer.txt
  elif [ "${1:-}" = update ]; then
    "$VENV/bin/pip" install -r requirements-trainer.txt
  fi
  PY="$VENV/bin/python"
}

setup_trainer() {                     # 환경이 없으면 만들고 $PY를 채운다
  if [ "$TRAINER_ENV" = auto ]; then
    if command -v conda >/dev/null; then TRAINER_ENV=conda; else TRAINER_ENV=venv; fi
  fi
  case "$TRAINER_ENV" in
    conda) setup_conda "${1:-}" ;;
    venv)  setup_venv "${1:-}" ;;
    *)     die "TRAINER_ENV는 auto|conda|venv 중 하나여야 합니다." ;;
  esac
  [ "$("$PY" -c 'import platform; print(platform.machine())')" = arm64 ] \
    || die "trainer Python이 arm64가 아닙니다(Rosetta). arm64용 conda/Python을 사용하세요."
}

proc_running() { [ -f "$RUN_DIR/$1.pid" ] && kill -0 "$(cat "$RUN_DIR/$1.pid")" 2>/dev/null; }

start_native() {                      # start_native <이름> <host> <port> <ASGI 앱>
  local name="$1" host="$2" port="$3" app="$4"
  if proc_running "$name"; then echo "ℹ️  $name 실행 중 (PID $(cat "$RUN_DIR/$name.pid"))"; return; fi
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  STORAGE_DIR="$ROOT/storage" KERAS_HOME="$ROOT/storage/keras" KERAS_BACKEND=tensorflow \
  TF_CPP_MIN_LOG_LEVEL=2 TRAINER_ENV="$TRAINER_ENV" TRAINER_URL="http://127.0.0.1:8001" \
  PYTHONPATH="$ROOT" \
    nohup "$PY" -m uvicorn "$app" --host "$host" --port "$port" >>"$LOG_DIR/$name.log" 2>&1 &
  echo $! > "$RUN_DIR/$name.pid"
}

stop_native() {                       # SIGTERM → trainer가 실행 중 Job도 정리한 뒤 종료
  local name="$1" pid
  if proc_running "$name"; then
    pid="$(cat "$RUN_DIR/$name.pid")"
    kill "$pid"
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid"; fi
    echo "⏹  $name 종료"
  fi
  rm -f "$RUN_DIR/$name.pid"
}

case "${1:-up}" in
  up)
    [ -f .env ] || cp .env.example .env
    mkdir -p storage/{keras,datasets,models,jobs,logs}
    if [ "$WEB_MODE" = native ] && [ "$TRAINER_MODE" = docker ]; then
      die "WEB_MODE=native는 TRAINER_MODE=native와 함께만 쓸 수 있습니다."
    fi
    if [ "$WEB_MODE" = docker ] || [ "$TRAINER_MODE" = docker ]; then check_docker; fi
    if [ "$TRAINER_MODE" = native ]; then
      setup_trainer
      if ! proc_running trainer; then
        start_native trainer "$TRAINER_HOST" 8001 app.trainer.main:app
        caffeinate -i -w "$(cat "$RUN_DIR/trainer.pid")" >/dev/null 2>&1 &   # trainer가 살아 있는 동안 Mac 유휴 잠자기 방지
      else
        echo "ℹ️  trainer 실행 중 (PID $(cat "$RUN_DIR/trainer.pid"))"
      fi
      wait_for "$TRAINER_HEALTH" trainer
      curl -s "$TRAINER_HEALTH"; echo                        # 학습 디바이스·환경·버전 출력
    fi
    if [ "$WEB_MODE" = docker ]; then
      echo "ℹ️  처음 실행이면 Docker Desktop이 프로젝트 폴더(Documents) 공유 승인을 묻습니다. 허용해야 web 컨테이너가 시작됩니다."
    fi
    if [ "$WEB_MODE" = native ]; then
      start_native web 127.0.0.1 8000 app.web.main:app      # trainer와 같은 Python 환경 사용
    elif [ "$TRAINER_MODE" = native ]; then
      docker compose up -d --build web
    else                                                     # 대체 경로 2: Docker CPU trainer
      TRAINER_URL=http://trainer:8001 docker compose --profile docker-trainer up -d --build
    fi
    wait_for "$WEB_URL/api/health" web
    echo "✅ $WEB_URL  (web: $WEB_MODE, trainer: $TRAINER_MODE)"
    [ -n "${NO_OPEN:-}" ] || open "$WEB_URL"            # NO_OPEN=1이면 브라우저를 열지 않음
    ;;
  setup)
    setup_trainer update
    echo "✅ trainer 환경 준비 완료 ($TRAINER_ENV: $PY)"
    ;;
  down)
    if docker_ready; then docker compose --profile docker-trainer down; fi
    stop_native web
    stop_native trainer
    ;;
  status)
    if docker_ready; then docker compose --profile docker-trainer ps; fi
    for n in trainer web; do
      if proc_running "$n"; then echo "$n(native): 실행 중 (PID $(cat "$RUN_DIR/$n.pid"))"; fi
    done
    curl -s "$TRAINER_HEALTH" || echo "trainer: 응답 없음"
    echo
    curl -s -o /dev/null -w "web: HTTP %{http_code}\n" "$WEB_URL/api/health" || echo "web: 응답 없음"
    ;;
  logs)
    name="${2:-trainer}"
    if [ -f "$LOG_DIR/$name.log" ] && { proc_running "$name" || ! docker_ready; }; then
      tail -n 100 -f "$LOG_DIR/$name.log"
    elif [ "$name" = web ]; then
      docker compose logs -f --tail=100 web
    else
      docker compose --profile docker-trainer logs -f --tail=100 trainer
    fi
    ;;
  reset)
    read -r -p "storage/models, storage/jobs를 모두 삭제합니다. 계속할까요? [y/N] " a
    if [ "$a" = y ]; then rm -rf storage/models storage/jobs && echo "삭제 완료"; fi
    ;;
  *) die "사용법: ./run.sh [up|down|status|logs [trainer|web]|setup|reset]" ;;
esac
