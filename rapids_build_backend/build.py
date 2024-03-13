# Copyright (c) 2024, NVIDIA CORPORATION.

from . import impls
from .config import Config
from .impls import (
    _get_backend,
    build_sdist,
    build_wheel,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
)

__all__ = [
    "build_wheel",
    "build_sdist",
    "_get_backend",
    "get_requires_for_build_wheel",
    "get_requires_for_build_sdist",
]

# The full set of hooks supported by build backends is enumerated in PEP517 and PEP 660.
# We unconditionally support a superset of the required hooks but do not
# unconditionally define all the optional hooks based on the following logic:
# - build_wheel and build_sdist are required (which also means every wrapped backend
#   will support them).
# - get_requires_for_build_wheel and get_requires_for_build_sdist are optional, but we
#   can provide safe default implementations because we always need to install any
#   requirements specified in the tool.rapids_build table even if the wrapped backend
#   doesn't add any.
# - build_editable is optional, and we can't provide a safe default implementation
#   because we won't support editable installs unless the wrapped backend does and the
#   mere existence of the hook will change the behavior of the build frontend to think
#   that the backend supports editable installs when rapids-build-backend only supports
#   them if the wrapped backend does.
# - prepare_metadata_for_build_wheel and prepare_metadata_for_build_editable are
#   optional, and we can't provide safe default implementations because we won't create
#   a dist-info directory unless the wrapped backend does it for us.
# - get_requires_for_build_editable is optional. We could provide a safe default
#   implementation, but there appears to be no guarantee that this won't change the
#   behavior of the frontend even if build_editable is not defined, so it is safer to
#   avoid defining it unless the wrapped backend does.

# We could override __getattr__ instead to defer these lookups, but in practice the
# result is equivalent since the underlying backend must be available when this module
# is imported. Any code that wishes to call or inspect the functions directly for any
# reason in a different environment may do so by importing the impls module directly.
config = Config()
backend = _get_backend(config.build_backend)
for name in (
    "build_editable",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
):
    if hasattr(backend, name):
        globals()[name] = getattr(impls, name)
        __all__.append(name)
