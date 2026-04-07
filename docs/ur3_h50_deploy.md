# UR3 h50 Deploy

이 문서는 현재 리포 기준으로 `pi05_ur3_pvi_dinov2_h50` / `pi05_ur3_pvi_hpr_h50` 체크포인트를 UR3에 연결된 PC로 옮기고 실행하는 절차를 정리한 것이다.

기준 경로:

- 소스 머신 repo: `/data/jhshin/openpi`
- 타깃 UR3 PC repo: `/data/jhshin/openpi`
- Hugging Face cache: `/data/jhshin/openpi/.cache/huggingface`

## 1. 소스 머신에서 파일 복사

권장 방식은 배포 스크립트를 쓰는 것이다. 이 스크립트는 UR3 실행에 필요한 최소 경로만 `rsync` 한다.

### DINO h50

```bash
cd /data/jhshin/openpi

./scripts/deploy_ur3_eval.sh \
  --target <USER>@<UR3_PC>:/data/jhshin/openpi \
  --variant dinov2_h50 \
  --copy-hf-cache
```

### HPR h50

```bash
cd /data/jhshin/openpi

./scripts/deploy_ur3_eval.sh \
  --target <USER>@<UR3_PC>:/data/jhshin/openpi \
  --variant hpr_h50 \
  --copy-hf-cache
```

`--copy-hf-cache`는 타깃 PC가 외부 인터넷 없이도 `facebook/dinov2-base`를 바로 읽을 수 있게 해준다. HPR도 내부적으로 DINOv2 backbone을 쓰므로 같이 복사하는 편이 안전하다.

스크립트가 복사하는 핵심 경로:

- `src/openpi`
- `examples/ur3`
- `_external/gello_software`
- `assets/pi05_ur3_pvi`
- `checkpoints/pi05_ur3_pvi_dinov2_h50/pi05_ur3_pvi_dinov2_h50_run1/1600`
- `checkpoints/pi05_ur3_pvi_hpr_h50/pi05_ur3_pvi_hpr_h50_run1/1600`
- `hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt` (HPR만)
- `packages/openpi-client`
- `pyproject.toml`, `uv.lock`, `.python-version`

## 2. UR3 PC에서 환경 준비

타깃 PC에서:

```bash
cd /data/jhshin/openpi

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install pyrealsense2
```

PyTorch 추론 경로는 `transformers` 패치가 필요하다.

```bash
cd /data/jhshin/openpi

SITE_PACKAGES=$(uv run python -c 'import site; print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))')
cp -r ./src/openpi/models_pytorch/transformers_replace/* "${SITE_PACKAGES}/transformers/"
```

## 3. 배포 검증

### DINO h50 체크

```bash
cd /data/jhshin/openpi

test -f checkpoints/pi05_ur3_pvi_dinov2_h50/pi05_ur3_pvi_dinov2_h50_run1/1600/model.safetensors
test -f checkpoints/pi05_ur3_pvi_dinov2_h50/pi05_ur3_pvi_dinov2_h50_run1/1600/assets/ur3_dataset/norm_stats.json
test -d _external/gello_software
test -d .cache/huggingface
```

### HPR h50 체크

```bash
cd /data/jhshin/openpi

test -f checkpoints/pi05_ur3_pvi_hpr_h50/pi05_ur3_pvi_hpr_h50_run1/1600/model.safetensors
test -f checkpoints/pi05_ur3_pvi_hpr_h50/pi05_ur3_pvi_hpr_h50_run1/1600/assets/ur3_dataset/norm_stats.json
test -f hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt
test -d _external/gello_software
test -d .cache/huggingface
```

RealSense가 제대로 잡히는지도 먼저 확인하는 편이 좋다.

```bash
cd /data/jhshin/openpi

HF_HOME=/data/jhshin/openpi/.cache/huggingface \
uv run examples/ur3/main.py \
  --policy-config pi05_ur3_pvi_dinov2_h50_infer \
  --policy-dir ./checkpoints/pi05_ur3_pvi_dinov2_h50/pi05_ur3_pvi_dinov2_h50_run1/1600 \
  --robot-mode direct \
  --robot-ip 192.168.5.102 \
  --base-camera-serial 335222074820 \
  --wrist-camera-serial 335522070336 \
  --mock \
  --max-steps 1
```

`--mock`는 정책 로딩과 CLI 경로만 먼저 확인할 때 유용하다. 실제 로봇 연결과 카메라 스트림을 같이 검증하려면 `--mock`를 빼고 짧게 실행한다.

## 4. DINO h50 실행

```bash
cd /data/jhshin/openpi

HF_HOME=/data/jhshin/openpi/.cache/huggingface \
uv run examples/ur3/main.py \
  --policy-config pi05_ur3_pvi_dinov2_h50_infer \
  --policy-dir ./checkpoints/pi05_ur3_pvi_dinov2_h50/pi05_ur3_pvi_dinov2_h50_run1/1600 \
  --robot-mode direct \
  --robot-ip 192.168.5.102 \
  --base-camera-serial 335222074820 \
  --wrist-camera-serial 335522070336 \
  --hz 30 \
  --replan-steps 8 \
  --debug-action-stats \
  --debug-log-every 1 \
  --default-prompt "pick up the pear and place it in the sink" \
  --max-steps 5000 \
  --deadband 0.003 \
  --kp 10 \
  --max-joint-velocity 0.35 \
  --max-joint-accel 0.8 \
  --speedj-accel 0.8 \
  --chunk-execution chunk_endpoint \
  --target-smoothing-alpha 0.2
```

## 5. HPR h50 실행

HPR h50는 아래 두 개만 바뀐다.

- `--policy-config pi05_ur3_pvi_hpr_h50_infer`
- `--policy-dir ./checkpoints/pi05_ur3_pvi_hpr_h50/pi05_ur3_pvi_hpr_h50_run1/1600`

전체 명령은 다음과 같다.

```bash
cd /data/jhshin/openpi

HF_HOME=/data/jhshin/openpi/.cache/huggingface \
uv run examples/ur3/main.py \
  --policy-config pi05_ur3_pvi_hpr_h50_infer \
  --policy-dir ./checkpoints/pi05_ur3_pvi_hpr_h50/pi05_ur3_pvi_hpr_h50_run1/1600 \
  --robot-mode direct \
  --robot-ip 192.168.5.102 \
  --base-camera-serial 335222074820 \
  --wrist-camera-serial 335522070336 \
  --hz 30 \
  --replan-steps 8 \
  --debug-action-stats \
  --debug-log-every 1 \
  --default-prompt "pick up the pear and place it in the sink" \
  --max-steps 5000 \
  --deadband 0.003 \
  --kp 10 \
  --max-joint-velocity 0.35 \
  --max-joint-accel 0.8 \
  --speedj-accel 0.8 \
  --chunk-execution chunk_endpoint \
  --target-smoothing-alpha 0.2
```

## 6. 문제 생기면 먼저 볼 것

- `ModuleNotFoundError: pyrealsense2`
  - `uv pip install pyrealsense2`
- `transformers_replace is not installed correctly`
  - `cp -r ./src/openpi/models_pytorch/transformers_replace/* "${SITE_PACKAGES}/transformers/"`
- `Norm stats not found`
  - `1600/assets/ur3_dataset/norm_stats.json`가 타깃에 있는지 확인
- HPR에서 보조 인코더 로딩 실패
  - `hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt` 존재 여부 확인
- 카메라 자동 탐지가 흔들림
  - 지금처럼 `--base-camera-serial`, `--wrist-camera-serial`를 명시
