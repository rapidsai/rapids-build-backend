#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

source rapids-date-string

python -m pip wheel . -w dist -vv --no-deps --disable-pip-version-check

RAPIDS_PY_WHEEL_NAME="rapids-build-backend" rapids-upload-wheels-to-s3 dist

# Run tests
python -m pip install $(ls dist/*.whl)[test]
python -m pytest -v tests/
