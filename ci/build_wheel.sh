#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

python -m pip wheel . -w dist -vv --no-deps --disable-pip-version-check

# Run tests
WHL_FILE=$(ls dist/*.whl)
python -m pip install "${WHL_FILE}[test]"
python -m pytest -v tests/

rapids-upload-wheels-to-s3 dist
