# CIFAR-100 CNN 학습·평가·웹캠 테스트 Web UI (PoC)

[Keras 예제 "Image classification from scratch"](https://keras.io/examples/vision/image_classification_from_scratch/)의 `make_model`을 32×32 입력에 맞게 고친 **xception_mini** 하나로 [CIFAR-100](https://keras.io/api/datasets/cifar100/)을 학습하고, 브라우저에서 데이터셋 선택·학습·모델 관리·평가·웹캠 테스트를 하는 PoC입니다.
설계 근거와 범위는 [docs/PoC_Proposal_CIFAR100_CNN_WebUI.md](docs/PoC_Proposal_CIFAR100_CNN_WebUI.md)를 보세요.

## 구성

```
브라우저 ──▶ web  (Docker, 127.0.0.1:8000)  ──HTTP──▶  trainer (macOS 네이티브 conda, 127.0.0.1:8001)
              UI · REST API · 웹캠 추론(CPU)            Job 큐(한 번에 1개) · 학습/평가 subprocess · Metal GPU
                      └──────────── ./storage (데이터셋 캐시 · 모델 · Job 상태 파일) ────────────┘
```

| 화면 | 기능 |
|---|---|
| 대시보드 `/` | web·trainer 연결, 학습 디바이스(GPU/CPU), 가용 메모리, 버전 일치 여부, 실행 중 작업, 최근 모델 |
| 데이터셋 `/datasets` | CIFAR-100 fine(100) / coarse(20) 준비(다운로드), 클래스별 샘플·분포 |
| 학습 `/train` | 하이퍼파라미터 입력, 빠른 데모 설정, 실시간 진행률·loss/acc 그래프·로그, 취소 |
| 모델 `/models` | 목록·상세·이름/메모/태그 수정·기본 모델 지정·삭제·`.keras` 다운로드, 2~3개 비교 |
| 평가 `/evaluate` | test 10,000장: top-1/top-5(95% CI), macro P/R/F1, confusion matrix, calibration(ECE), 클래스별 지표, 오분류 샘플, McNemar 검정 |
| 테스트 `/test` | MacBook 웹캠 실시간 top-5 예측 (중앙 ROI, 32×32 입력 미리보기, 5프레임 평균, 스냅샷) |

모든 화면은 **한국어 · English · 中文 · Tiếng Việt**를 지원합니다 (아래 [화면 언어](#화면-언어)).

## 준비 (팀원별 1회)

1. **Docker Desktop**을 실행하고 첫 실행 안내(이용약관 등)를 마칩니다. 첫 `./run.sh up` 때 프로젝트 폴더 공유 승인을 묻는 창이 뜨면 허용합니다. Settings → Resources에서 **Memory 2GB, Swap 1GB**를 권장합니다.
2. **arm64용 conda** (권장: Miniforge `brew install --cask miniforge`, 이미 Anaconda/Miniconda가 있으면 `conda info`의 `platform : osx-arm64` 확인)
3. 나머지(conda 환경 `cnn-poc-trainer` 생성, 패키지 설치)는 `./run.sh up`이 처음 실행할 때 자동으로 합니다.

## 실행

```bash
./run.sh up
```

브라우저에서 http://localhost:8000 이 열립니다. 처음 쓸 때는 **데이터셋 → 준비**를 눌러 CIFAR-100(약 161MB)을 내려받으세요(학습을 바로 시작해도 자동으로 내려받습니다).

```bash
./run.sh status          # 상태 확인
./run.sh logs trainer    # trainer 로그 (logs web: web 로그)
./run.sh down            # web 컨테이너와 trainer 프로세스 모두 종료
./run.sh setup           # 의존성 파일을 바꾼 뒤 conda 환경 갱신
./run.sh reset           # storage/models, storage/jobs 삭제 (확인 후)
```

### 실행 옵션 (환경 변수)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `TRAINER_MODE` | `native` | `docker`: Docker CPU trainer (대체 경로 2, Docker 메모리 4GB 필요) |
| `TRAINER_ENV` | `auto` | `conda` / `venv` (conda가 없으면 auto가 venv 선택) |
| `TRAINER_DEVICE` | `auto` | `cpu`: Metal을 쓰지 않음 (대체 경로 1) / `gpu`: GPU가 없으면 실패 |
| `WEB_MODE` | `docker` | `native`: web도 conda 환경에서 실행 (Docker 없이 점검·개발용) |
| `TRAINER_HOST`, `TRAINER_TOKEN` | `127.0.0.1`, 없음 | 컨테이너에서 trainer 연결이 안 될 때 `0.0.0.0` + 토큰 |
| `MIN_FREE_MEM_GB` | `1.0` | 가용 메모리가 이보다 적으면 Job 시작을 미룸 |
| `NO_OPEN` | 없음 | `1`이면 up 후 브라우저를 열지 않음 |

예: Docker 없이 전체 기능 확인

```bash
WEB_MODE=native ./run.sh up
```

## 화면 언어

상단바 오른쪽 🌐 선택 상자에서 바꿉니다. 고른 언어는 브라우저 쿠키(`lang`)에 1년 동안 저장되어 팀원마다 따로 유지됩니다. 처음 접속하면 브라우저 언어 설정을 따르고, 지원하지 않는 언어면 한국어로 보입니다.

- 번역 사전: `app/core/locales/{ko,en,zh,vi}.json`. 문구를 고칠 때는 네 파일을 함께 고친 뒤 web을 다시 시작합니다(`./run.sh down` → `./run.sh up`).
- 점검: `python3 -m app.core.i18n` 이 언어별 키·자리표시자 일치, 코드에서 쓰는 키 누락 여부를 확인합니다.
- 번역하지 않는 것: CIFAR-100 클래스 이름(데이터셋 원래 레이블), 직접 입력한 모델 이름·메모·태그, 학습 로그, 평가 그림(영어), epoch·top-1·ECE 같은 전문 용어.
- 자세한 구조: 제안서 §9.4

## 고정한 버전 (Phase 0 확인)

`tensorflow-metal 1.2.0`이 지원하는 **TensorFlow 2.18.1**에 맞췄고, web 컨테이너(Linux)도 같은 버전을 씁니다. ★ 패키지는 web·trainer가 반드시 같아야 하며, 다르면 대시보드에 경고가 뜹니다.

| 패키지 | 버전 |
|---|---|
| Python | 3.11 |
| tensorflow ★ | 2.18.1 (+ tensorflow-metal 1.2.0, macOS만) |
| keras ★ | 3.15.1 |
| numpy ★ | 2.0.2 |

의존성 파일: `environment-trainer.yml`(conda), `requirements-common.txt`, `requirements-trainer.txt`(macOS), `requirements-docker.txt`(Linux 컨테이너)

## Phase 0 실측 결과 (MacBook M2 8GB)

| 항목 | 결과 |
|---|---|
| epoch당 시간 (전체 45,000장) | w=0.5: GPU 약 1.5분 / CPU 약 2.8분 · w=1.0: GPU 약 3.2분 / CPU 약 5.8분 |
| 빠른 데모 설정 (w=0.5, 9,000장, 3 epoch) | GPU 약 1분, val acc 13.7% |
| w=0.5, 전체 데이터, 10 epoch | GPU 13.5분, **test top-1 39.4%** (95% CI 38.4~40.3%), top-5 70.3% |
| 웹캠 | 400ms 간격에서 2.2fps (왕복 약 45ms) |

예제와 다른 점 하나: BatchNormalization momentum을 0.99(기본) → **0.9**로 바꿨습니다. 0.99로는 짧은 학습에서 추론 예측이 한 클래스로 몰렸습니다(제안서 §5.1, §12.3).

## 폴더 구조

```
app/core/      데이터셋 레지스트리, xception_mini, 학습·평가·준비 스크립트, 모델 레지스트리, 통계, 다국어(i18n.py, locales/)
app/trainer/   trainer 서비스 (FastAPI, Job 큐, subprocess 관리)
app/web/       web 서비스 (FastAPI, Jinja2 화면, 정적 JS/CSS, 웹캠 추론)
storage/       keras/(데이터셋 캐시) datasets/ models/ jobs/ logs/  ← git 제외
```

모델 파일: `storage/models/{model_id}/model.keras` (+ `meta.json`, `history.csv`, `summary.txt`, `evaluations/`)

## 문제 해결

| 증상 | 확인 |
|---|---|
| web 컨테이너가 `Created`에서 멈추고 `up`이 끝나지 않음 | 첫 실행 시 Docker Desktop이 프로젝트 폴더(Documents) **파일 공유 승인**을 묻습니다. Docker Desktop 창(또는 macOS 알림)에서 허용하세요 |
| `Docker Desktop을 먼저 실행하세요` | Docker Desktop을 켜고 고래 아이콘이 Running이 될 때까지 기다린 뒤 다시 실행. 급하면 `WEB_MODE=native ./run.sh up` |
| 대시보드에 `trainer ● 중지됨` | `./run.sh logs trainer`. Docker web에서만 안 되면 `TRAINER_HOST=0.0.0.0 TRAINER_TOKEN=<임의 문자열> ./run.sh up` |
| 학습 디바이스가 `CPU` | `conda run -n cnn-poc-trainer python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` |
| Job이 "가용 메모리 부족 … 대기 중" | 브라우저 탭·다른 앱을 닫거나 `MIN_FREE_MEM_GB=0.5 ./run.sh up` |
| 웹캠이 안 열림 | `http://localhost:8000`으로 접속했는지, macOS 시스템 설정 → 개인정보 보호 및 보안 → 카메라에서 브라우저 허용 |
| 학습 중 Mac이 잠듦 | `run.sh`가 `caffeinate`로 유휴 잠자기를 막지만, 덮개를 닫으면 잠듭니다 |
