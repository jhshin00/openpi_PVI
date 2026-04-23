#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROBOTWIN_DIR="${ROOT_DIR}/third_party/robotwin"
CUROBO_COMMIT="cca894de9ec74e77a0a4071319c81958991a9108"

PYTHON_VERSION="3.11"
VENV_DIR="${SCRIPT_DIR}/.venv"
DOWNLOAD_ASSETS=0
INSTALL_CUROBO=1
INSTALL_PYTORCH3D=1

usage() {
  cat <<EOF
Usage: bash examples/robotwin/bootstrap_eval_env.sh [options]

Options:
  --python <version>      Python version for the eval env. Default: 3.11
  --venv-dir <path>       Virtual environment directory. Default: examples/robotwin/.venv
  --download-assets       Run RoboTwin asset download after package install
  --skip-curobo           Skip CuRobo installation
  --skip-pytorch3d        Skip PyTorch3D installation
  -h, --help              Show this help

This script follows the RoboTwin official Pi0.5 flow:
1. create a dedicated uv virtualenv
2. install openpi into that env
3. install RoboTwin simulator dependencies
4. apply the local transformers replacement modules
5. install R3M with no dependency resolution
6. install PyTorch3D
7. apply the official sapien/mplib patches
8. install the RoboTwin-compatible CuRobo revision
9. optionally download RoboTwin assets
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --download-assets)
      DOWNLOAD_ASSETS=1
      shift
      ;;
    --skip-curobo)
      INSTALL_CUROBO=0
      shift
      ;;
    --skip-pytorch3d)
      INSTALL_PYTORCH3D=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -d "${ROBOTWIN_DIR}" ]]; then
  echo "RoboTwin submodule not found at ${ROBOTWIN_DIR}" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Warning: ffmpeg was not found in PATH. RoboTwin Pi0.5 docs require a working ffmpeg install." >&2
fi

mkdir -p "$(dirname "${VENV_DIR}")"

echo "[1/9] Creating eval env at ${VENV_DIR} with Python ${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "[2/9] Installing openpi into the active eval env"
GIT_LFS_SKIP_SMUDGE=1 uv sync --project "${ROOT_DIR}" --active --frozen --no-dev

echo "[3/9] Installing RoboTwin simulator extras"
uv pip install -r "${SCRIPT_DIR}/requirements-eval.txt"
uv pip install "pytest>=8.3.4" "hydra-core>=1.3,<1.4"

echo "[4/9] Applying transformers_replace to the active eval env"
python - <<'PY'
from pathlib import Path
import shutil
import transformers

source_dir = Path("src/openpi/models_pytorch/transformers_replace").resolve()
target_dir = Path(transformers.__file__).resolve().parent
for path in source_dir.iterdir():
    destination = target_dir / path.name
    if path.is_dir():
        shutil.copytree(path, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(path, destination)
print(f"Copied transformers_replace from {source_dir} to {target_dir}")
PY
python - <<'PY'
import transformers
from transformers.models.siglip import check

if transformers.__version__ != "4.53.2":
    raise RuntimeError(f"Expected transformers==4.53.2, got {transformers.__version__}")
if not check.check_whether_transformers_replace_is_installed_correctly():
    raise RuntimeError("transformers_replace check failed")
print("transformers_replace check ok")
PY

echo "[5/9] Installing R3M without transitive dependencies"
uv pip install "git+https://github.com/facebookresearch/r3m.git" --no-deps

if [[ "${INSTALL_PYTORCH3D}" -eq 1 ]]; then
  echo "[6/9] Installing PyTorch3D"
  uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
else
  echo "[6/9] Skipping PyTorch3D installation"
fi

echo "[7/9] Applying RoboTwin patches to sapien and mplib"
python - <<'PY'
from pathlib import Path
import inspect
import mplib
import sapien


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"Already patched: {path}")
        return
    if old not in text:
        raise RuntimeError(f"Could not find expected text in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"Patched: {path}")


sapien_root = Path(inspect.getfile(sapien)).resolve().parent
urdf_loader = sapien_root / "wrapper" / "urdf_loader.py"
replace_once(
    urdf_loader,
    'with open(urdf_file, "r") as f:',
    'with open(urdf_file, "r", encoding="utf-8") as f:',
)
replace_once(
    urdf_loader,
    'srdf_file = urdf_file[:-4] + "srdf"',
    'srdf_file = urdf_file[:-4] + ".srdf"',
)
replace_once(
    urdf_loader,
    'with open(srdf_file, "r") as f:',
    'with open(srdf_file, "r", encoding="utf-8") as f:',
)

mplib_root = Path(inspect.getfile(mplib)).resolve().parent
planner = mplib_root / "planner.py"
replace_once(
    planner,
    "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
    "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
)
PY

if [[ "${INSTALL_CUROBO}" -eq 1 ]]; then
  echo "[8/9] Installing CuRobo at ${CUROBO_COMMIT}"
  CUROBO_DIR="${ROBOTWIN_DIR}/envs/curobo"
  if [[ ! -d "${CUROBO_DIR}/.git" ]]; then
    git clone https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
  fi
  git -C "${CUROBO_DIR}" fetch origin "${CUROBO_COMMIT}"
  git -C "${CUROBO_DIR}" checkout "${CUROBO_COMMIT}"
  uv pip uninstall nvidia-curobo curobo || true
  uv pip install -e "${CUROBO_DIR}" --no-build-isolation
  python - <<'PY'
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

print("CuRobo import check ok:", Pose, JointState, MotionGen, MotionGenConfig)
PY
else
  echo "[8/9] Skipping CuRobo installation"
fi

if [[ "${DOWNLOAD_ASSETS}" -eq 1 ]]; then
  echo "[9/9] Downloading RoboTwin assets"
  (
    cd "${ROBOTWIN_DIR}"
    bash script/_download_assets.sh
  )
else
  echo "[9/9] Skipping RoboTwin asset download. Run with --download-assets when ready."
fi

cat <<EOF

Eval env is ready: ${VENV_DIR}

Activate it with:
  source "${VENV_DIR}/bin/activate"

Typical next steps:
  1. Download RoboTwin assets if you skipped them:
       bash examples/robotwin/bootstrap_eval_env.sh --download-assets
  2. Run same-process RoboTwin eval:
       bash examples/robotwin/eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id>

This env is intended for RoboTwin simulator + openpi/pi0.5/PVI policy in one process.
EOF
