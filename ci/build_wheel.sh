#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# TODO: revert before merging (just trying to quiet down the noise to find the root cause)
rm /etc/xdg/pip/pip.conf

python -m pip wheel . -w dist -vv --no-deps --disable-pip-version-check

# Run tests
WHL_FILE=$(ls dist/*.whl)
python -m pip install "${WHL_FILE}[test]"
python -m pytest -v tests/

RAPIDS_PY_WHEEL_NAME="rapids-build-backend" rapids-upload-wheels-to-s3 dist
