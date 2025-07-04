# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Helper functions for configuring IREE and dependent project WORKSPACE files."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

CUDA_TOOLKIT_ROOT_ENV_KEY = "IREE_CUDA_TOOLKIT_ROOT"

# Our CI docker images use a stripped down CUDA directory tree in some
# images, and it is tailored just to support building key elements.
# When this is done, the IREE_CUDA_DEPS_DIR env var is set, and we
# respect that here in order to match the CMake side (which needs it
# because CUDA toolkit detection differs depending on whether it is
# stripped down or not).
# TODO: Simplify this on the CMake/docker side and update here to match.
# TODO(#15332): Dockerfiles no longer include these deps. Simplify.
CUDA_DEPS_DIR_FOR_CI_ENV_KEY = "IREE_CUDA_DEPS_DIR"

def cuda_auto_configure_impl(repository_ctx):
    env = repository_ctx.os.environ
    iree_repo_alias = repository_ctx.attr.iree_repo_alias
    cuda_toolkit_root = None

    script_file = repository_ctx.attr.script
    if script_file:
        # Only support Linux and Windows.
        # (Assuming repository_ctx.os.name returns a lowercase string like "linux" or "windows".)
        if repository_ctx.os.arch.lower() not in ["amd64"] or repository_ctx.os.name.lower() not in ["linux", "windows"]:
            repository_ctx.file("BUILD", """
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
package(default_visibility = ["//visibility:public"])
bool_flag(
    name = "enabled",
    build_setting_default = False,
)""")
            print("CUDA auto configuration downloading is only supported on x86_64 Linux and Windows, but the current platform is: " + repository_ctx.os.name + " " + repository_ctx.os.arch)
            return
        # Link and run the download script.
        script_path = repository_ctx.path(script_file)
        repository_ctx.symlink(script_path, "download.py")
        output_dir = repository_ctx.path("cuda_root")

        result = repository_ctx.execute(["python3", "download.py", output_dir], timeout=40)
        if result.return_code != 0:
            fail("download.py execution failed:\n" + result.stderr)
        
        cuda_toolkit_root = "cuda_root/12.2.1/linux-x86_64"
    else:
        # Probe environment for CUDA toolkit location.
        env_cuda_toolkit_root = env.get(CUDA_TOOLKIT_ROOT_ENV_KEY)
        env_cuda_deps_dir_for_ci = env.get(CUDA_DEPS_DIR_FOR_CI_ENV_KEY)
        if env_cuda_toolkit_root:
            cuda_toolkit_root = env_cuda_toolkit_root
        elif env_cuda_deps_dir_for_ci:
            cuda_toolkit_root = env_cuda_deps_dir_for_ci

    # If we found a CUDA toolkit root (either via download or the environment),
    # create symlinks for the directories and libdevice.
    libdevice_rel_path = "iree_local/libdevice.bc"
    if cuda_toolkit_root:
        repository_ctx.symlink(cuda_toolkit_root + "/include", "include")
        repository_ctx.symlink(
            cuda_toolkit_root + "/nvvm/libdevice/libdevice.10.bc",
            libdevice_rel_path,
        )

    repository_ctx.template(
        "BUILD",
        Label("%s//:build_tools/third_party/cuda/BUILD.template" % iree_repo_alias),
        {
            "%ENABLED%": "True" if cuda_toolkit_root else "False",
            "%LIBDEVICE_REL_PATH%": libdevice_rel_path if cuda_toolkit_root else "BUILD",
            "%IREE_REPO_ALIAS%": iree_repo_alias,
        },
    )

cuda_auto_configure = repository_rule(
    environ = [
        CUDA_DEPS_DIR_FOR_CI_ENV_KEY,
        CUDA_TOOLKIT_ROOT_ENV_KEY,
    ],
    implementation = cuda_auto_configure_impl,
    attrs = {
        "iree_repo_alias": attr.string(default = "@iree_core"),
        "script": attr.label(allow_files = True, default=None),
    },
    local = True
)

def configure_iree_cuda_deps(iree_repo_alias = None, script = ""):
    maybe(
        cuda_auto_configure,
        name = "iree_cuda",
        iree_repo_alias = iree_repo_alias,
        script = script,
    )

def configure_iree_submodule_deps(iree_repo_alias = "@", iree_path = "./"):
    """Configure all of IREE's simple repository dependencies that come from submodules.

    Simple is defined here as just calls to `local_repository` or
    `new_local_repository`. This assumes you have a directory that includes IREE
    and all its submodules. Note that fetching a GitHub archive does not include
    submodules.
    Yes it is necessary to have both the workspace alias and path argument...

    Args:
      iree_repo_alias: The alias for the IREE repository.
      iree_path: The path to the IREE repository containing submodules
    """

    maybe(
        native.local_repository,
        name = "com_google_googletest",
        path = paths.join(iree_path, "third_party/googletest"),
    )

    maybe(
        native.new_local_repository,
        name = "com_github_dvidelabs_flatcc",
        build_file = iree_repo_alias + "//:build_tools/third_party/flatcc/BUILD.overlay",
        path = paths.join(iree_path, "third_party/flatcc"),
    )

    maybe(
        native.new_local_repository,
        name = "vulkan_headers",
        build_file = iree_repo_alias + "//:build_tools/third_party/vulkan_headers/BUILD.overlay",
        path = paths.join(iree_path, "third_party/vulkan_headers"),
    )

    maybe(
        native.local_repository,
        name = "stablehlo",
        path = paths.join(iree_path, "third_party/stablehlo"),
    )

    maybe(
        native.local_repository,
        name = "com_google_benchmark",
        path = paths.join(iree_path, "third_party/benchmark"),
    )

    maybe(
        native.local_repository,
        name = "cpuinfo",
        path = paths.join(iree_path, "third_party/cpuinfo"),
    )

    maybe(
        native.new_local_repository,
        name = "spirv_cross",
        build_file = iree_repo_alias + "//:build_tools/third_party/spirv_cross/BUILD.overlay",
        path = paths.join(iree_path, "third_party/spirv_cross"),
    )

    maybe(
        native.new_local_repository,
        name = "tracy_client",
        build_file = iree_repo_alias + "//:build_tools/third_party/tracy_client/BUILD.overlay",
        path = paths.join(iree_path, "third_party/tracy"),
    )

    maybe(
        native.new_local_repository,
        name = "nccl",
        build_file = iree_repo_alias + "//:build_tools/third_party/nccl/BUILD.overlay",
        path = paths.join(iree_path, "third_party/nccl"),
    )

    maybe(
        native.new_local_repository,
        name = "hsa_runtime_headers",
        build_file = iree_repo_alias + "//:build_tools/third_party/hsa-runtime-headers/BUILD.overlay",
        path = paths.join(iree_path, "third_party/hsa-runtime-headers"),
    )

    maybe(
        native.new_local_repository,
        name = "webgpu_headers",
        build_file = iree_repo_alias + "//:build_tools/third_party/webgpu-headers/BUILD.overlay",
        path = paths.join(iree_path, "third_party/webgpu-headers"),
    )
