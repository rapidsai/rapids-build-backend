#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

python -m pip wheel . -w dist -vv --no-deps --disable-pip-version-check

# Run tests
python -m pip install $(ls dist/*.whl)[test]
python -m pytest -v tests/
