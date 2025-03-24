#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

wheel_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}

python -m pip wheel . -w "${wheel_dir}" -vv --no-deps --disable-pip-version-check

# Run tests
WHL_FILE=$(ls "${wheel_dir}"/*.whl)
python -m pip install "${WHL_FILE}[test]"
python -m pytest -v tests/

RAPIDS_PY_WHEEL_NAME="rapids-build-backend" rapids-upload-wheels-to-s3 "${wheel_dir}"
