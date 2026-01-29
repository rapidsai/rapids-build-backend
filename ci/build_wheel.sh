#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

python -m pip wheel -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" -vv --no-deps --disable-pip-version-check

ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

# Run tests
WHL_FILE=$(ls "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/*.whl)
python -m pip install "${WHL_FILE}[test]"
python -m pytest -v tests/
