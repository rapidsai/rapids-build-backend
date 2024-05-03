#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

UPLOAD_PACKAGES="${1:-false}"

PKG_DIR="${PWD}/conda_package"
rapids-conda-retry mambabuild --output-folder "${PKG_DIR}" conda/recipes/rapids-build-backend
