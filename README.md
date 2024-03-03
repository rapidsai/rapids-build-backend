# rapids PEP517 build backend

`rapids_builder` is an adapter around PEP517 builders that provides support for key RAPIDS requirements.
It currently support `scikit-build-core` and `setuptools` as the wrapped builder.
The package's primary purpose is to automate the CUDA versioning strategy for wheels.
It modifies the package name to include a suffix of the form `-cu${CUDA_VERSION}`, the standard in RAPIDS.
It also modifies build requirements to ensure that isolated builds have the necessary requirements, and it modifies the install requirements as well so that they are propagated to the built wheel.

## Supported configuration
The base configuration of `rapids_builder` is identical to the `build-system` table in pyproject.toml.
- The required `tool.rapids_builder.build-backend` key specifies the backend. In this case, `rapids_builder` will be the value of `build-system.build-backend`, while the wrapped builder should be set to the value of `tool.rapids_builder.build-backend`.
- The optional `tool.rapids_builder.requires` list specifies build requirements. Any known RAPIDS dependency in this list will be suffixed with a CUDA suffix. Other dependencies will be treated as if they were in `build-system.requires`.
- The optional `tool.rapids_builder.allow-no-cuda` flag indicates whether a package allows building on a system without nvcc. It defaults to False. If True, then no suffixes are added to the package name itself or any of its dependencies.

## TODO: We need to determine whether this behavior can always be on or if it needs to be configurable.
Some cases to consider:
- Building wheels: Here we definitely need to change everything (name, build dependencies, install dependencies).
- Installing (not building wheels): In this case the name doesn't matter because the wheel is ephemeral (the package files are immediately put into the site directory). If we do not specify --no-deps, then we do want the dependencies to be installed with the appropriate suffixes.
- Editable installs: Again, the wheel name doesn't matter here since the wheel is ephemeral (in this case we end up pointing back to the source dir or to build `so`s) so the rename is fine. The dependencies do need to be renamed as above.
- No build isolation: In this case the build dependencies don't matter. Otherwise everything is the same, regardless of whether we build wheels or install (editable or not).
- conda builds: conda builds should effectively be equivalent to `pip install --no-build-isolation --no-deps`, so none of the changes matter. If meta.yaml parses pyproject.toml for any data, it must happen before the `pip install` call. However, we should verify this to be safe.

The most sensible choice might be on by default, but with a switch to turn off.

## TODO: Determine how we want to split up build requirements between `build-system` and `tool.rapids_builder`

In theory any dependency that doesn't need suffixing could also go into `build-system.requires`.
I think it's easier to teach that all dependencies other than `rapids_builder` itself should to into `tool.rapids_builder`, but I don't know how others feel.

## Rejected ideas

- We could also include the rewrite of VERSION that we use for RAPIDS builds, but this is really more specific to our release process than the general process of building our wheels.
