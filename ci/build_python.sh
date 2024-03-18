#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

UPLOAD_PACKAGES="${1:-false}"

PKG_DIR="${PWD}/conda_package"
rapids-conda-retry mambabuild --output-folder "${PKG_DIR}" conda/recipes/rapids-build-backend

if [ "$UPLOAD_PACKAGES" = "true" ]; then
    # TODO: Figure out the best way to get CONDA_PKG_FILE
    rapids-retry anaconda -t "${RAPIDS_CONDA_TOKEN}" upload --label main --skip-existing --no-progress "${PKG_DIR}/noarch/"*.tar.bz2
fi
