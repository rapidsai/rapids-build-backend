# Copyright (c) 2024, NVIDIA CORPORATION.

import os
from typing import TYPE_CHECKING

from .utils import _get_pyproject

if TYPE_CHECKING:
    from typing import Any


class Config:
    """Manage the build configuration for the project."""

    # Mapping from config option to default value (None indicates that option is
    # required) and whether it may be overridden by an environment variable or a config
    # setting.
    config_options: "dict[str, tuple[Any, bool]]" = {
        "build-backend": (None, False),
        "commit-file": ("", False),
        "dependencies-file": ("dependencies.yaml", True),
        "disable-cuda-suffix": (False, True),
        "matrix-entry": ("", True),
        "require-cuda": (True, True),
        "requires": (lambda: [], False),
    }

    def __init__(self, dirname=".", config_settings=None):
        self.config_settings = config_settings or {}
        pyproject_data = _get_pyproject(dirname)
        try:
            self.config = pyproject_data["tool"]["rapids-build-backend"]
        except KeyError as e:
            raise RuntimeError("No rapids-build-backend table in pyproject.toml") from e

    def __getattr__(self, name):
        config_name = name.replace("_", "-")
        if config_name in Config.config_options:
            default_value, allows_override = Config.config_options[config_name]
            if callable(default_value):
                default_value = default_value()

            # If overrides are allowed environment variables take precedence over the
            # config_settings dict.
            if allows_override:
                if (env_var := f"RAPIDS_{name.upper()}") in os.environ:
                    # Anything overridable by an environment variable must have a
                    # default. The input is a string, but we need to convert it to the
                    # appropriate type, which we determine based on the default value.
                    if isinstance(default_value, bool):
                        str_val = os.environ[env_var]
                        if str_val not in ("true", "false"):
                            raise ValueError(
                                f"{env_var} must be 'true' or 'false', not {str_val}"
                            )
                        return str_val == "true"
                    return os.environ[env_var]

                if config_name in self.config_settings:
                    if isinstance(default_value, bool):
                        return self.config_settings[config_name] == "true"
                    return self.config_settings[config_name]

            try:
                return self.config[config_name]
            except KeyError:
                if default_value is not None:
                    return default_value

                raise AttributeError(f"Config is missing required attribute {name}")
        else:
            raise AttributeError(f"Attempted to access unknown option {name}")
