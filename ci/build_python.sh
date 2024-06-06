#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-conda-retry mambabuild conda/recipes/rapids-build-backend

rapids-upload-conda-to-s3 python
