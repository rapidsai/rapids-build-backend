# RAPIDS PEP517 build backend

`rapids-build-backend` is an adapter around PEP517 builders that provides support for key RAPIDS requirements.

The package's primary purpose is to automate the various bits of preprocessing that are typically done to RAPIDS package metadata prior to publishing packages.

It is responsible for the following:

- Determining the correct dependencies for the package, based on the target CUDA version and architecture
  - *by running [`rapids-dependency-file-generator`](https://github.com/rapidsai/dependency-file-generator)*
- Modifying the package name to include a CUDA suffix (e.g. `"rmm" -> "rmm-cu12"`)
- Updating the git commit embedded in the importable package.

## Supported builders

The project is known to support the following builders:

* `scikit-build-core`
* `setuptools`

## Supported configuration

`rapids-build-backend` exposes configuration for skipping or modifying behaviors that might be desirable only in some scenarios (e.g. wheel vs conda builds vs editable installs).

Any option without a default is required.

| Option                | Definition                                                                                       | Type           | Default                       | Supports dynamic modification |
|-----------------------|--------------------------------------------------------------------------------------------------|----------------|-------------------------------|-------------------------------|
| `build-backend`       | The wrapped build backend (e.g. `setuptools.build_meta`)                                         | string         |                               | N                             |
| `commit-files`        | List of files in which to write the git commit hash                                              | list[str]      | ["<project_name>/GIT_COMMIT"] | N                             |
| `dependencies-file`   | The path to the `dependencies.yaml` file to use                                                  | string         | "dependencies.yaml"           | Y                             |
| `disable-cuda`        | If true, CUDA version in build environment is ignored when setting package name and dependencies | bool           | false                         | Y                             |
| `matrix-entry`        | A `;`-separated list of `=`-delimited key/value pairs                                            | string         | ""                            | Y                             |
| `requires`            | List of build requirements (in addition to `build-system.requires`)                              | list[str]      | []                            | N                             |

This configuration can be provided via the following mechanisms:

* `[tool.rapids-build-backend]` table in `pyproject.toml`
* `-C / --config-settings` passed to tools like `build` and `pip`
   - *(prefixed with `rapidsai.`, e.g. `pip wheel -C rapidsai.disable=cuda=true .`)*
* environment variables
    - *(prefixed with `RAPIDS_`, e.g. `RAPIDS_DISABLE_CUDA=true pip wheel .`)*

## `setuptools` support

This project supports builds using `setuptools.build_meta` as their build backend, and which use a `setup.py` for configuration.

However, it does not support passing a list of dependencies through `setup_requires` to `setuptools.setup()`.
If you're interested in using `setuptools.build_meta` and a `setup.py`, pass a list of dependencies that need to be installed prior to `setup.py` running through `rapids-build-backend`'s requirements, like this:

```toml
[project]
build-backend = "rapids_build_backend.build"
requires = [
    "rapids-build-backend",
    "setuptools"
]

[tool.rapids-build-backend]
build-backend = "setuptools.build_meta"
requires = [
    "Cython"
]
```

## Other build dependencies

When using `rapids-build-backend`, the `[build-system]` table in `pyproject.toml` should only include `rapids-build-backend` and the library providing the build backend it wraps.

For example:

```toml
[build-system]
build-backend = "rapids_build_backend.build"
requires = [
    "rapids-build-backend>=0.3.0,<0.4.0dev0",
    "setuptools>=64.0.0",
]
```

Any other build-time dependencies should be provided via `requires` in the `[tool.rapids-build-backend]` table.

For example:

```toml
[tool.rapids-build-backend]
build-backend = "setuptools.build_meta"
dependencies-file = "dependencies.yaml"
requires = [
    "cython>=3.0.0",
]
```
