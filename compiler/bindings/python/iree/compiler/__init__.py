# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE compiler Python bindings."""

# Re-export some legacy APIs from the tools package to this top-level.
# TODO: Deprecate and remove these names once clients are migrated.
from .tools import *

# Roofline AI ->
import sys

if sys.version_info[:2] >= (3, 8):
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Note that the package name must be kept in sync with
    # roof-iree/runtime/BUILD.bazel
    __version__ = version("iree_compiler_bazel")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
# <- Roofline AI
