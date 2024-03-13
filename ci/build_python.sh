#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-date-string

rapids-print-env

rapids-logger "Begin py build"

rapids-conda-retry mambabuild conda/recipes/rapids-build-backend

rapids-upload-conda-to-s3 python
