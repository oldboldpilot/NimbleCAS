#!/usr/bin/env bash
# Create/refresh the uv-managed Python environment for the nanobind bindings.
# @author Olumuyiwa Oluwasanmi
#
# All Python dependencies are managed with uv from pyproject.toml. This creates the
# project .venv and installs the locked dependencies (writing uv.lock).

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_common.sh"

UV="${UV:-$(command -v uv || true)}"
if [[ -z "${UV}" && -x "${HOME}/.local/bin/uv" ]]; then
    UV="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV}" || ! -x "${UV}" ]]; then
    echo "error: uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

cd "${REPO_ROOT}"
# uv sync creates .venv (if needed) and installs the pyproject dependencies,
# writing uv.lock for reproducibility.
"${UV}" sync
echo "uv env ready: ${REPO_ROOT}/.venv"
