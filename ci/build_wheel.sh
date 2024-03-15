#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

UPLOAD_PACKAGES="${1:-false}"

python -m pip wheel . -w dist -vv --no-deps --disable-pip-version-check

# Run tests
WHL_FILE=$(ls dist/*.whl)
python -m pip install "${WHL_FILE}[test]"
python -m pytest -v tests/

if [ "$UPLOAD_PACKAGES" = "true" ]; then
    anaconda -t "${RAPIDS_CONDA_TOKEN}" upload --skip-existing --no-progress ${WHL_FILE}
fi
