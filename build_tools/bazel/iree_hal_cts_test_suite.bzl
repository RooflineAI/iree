load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_test")
load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")
load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("//build_tools/bazel:cts_test_gen.bzl", "cts_test_gen")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

CTS_ALL_TESTS = [
    "allocator",
    "buffer_mapping",
    "command_buffer",
    "command_buffer_copy_buffer",
    "command_buffer_dispatch",
    "command_buffer_dispatch_constants",
    "command_buffer_fill_buffer",
    "command_buffer_update_buffer",
    "driver",
    "event",
    "executable_cache",
    "file",
    "semaphore",
    "semaphore_submission",
]

CTS_EXECUTABLE_TESTS = [
    "command_buffer_dispatch",
    "command_buffer_dispatch_constants",
    "executable_cache",
]

CTS_EXECUTABLE_SOURCES = [
    "command_buffer_dispatch_test",
    "command_buffer_dispatch_constants_test",
    "executable_cache_test",
]

def iree_hal_cts_test_suite(
        name,
        driver_name,
        driver_registration_hdr,
        driver_registration_fn,
        deps = [],
        variant_suffix = None,
        compiler_target_backend = None,
        compiler_target_device = None,
        compiler_flags = [],
        executable_format = None,
        included_tests = None,
        excluded_tests = [],
        args = [],
        tags = [],
        **kwargs):
    """Creates a set of CTS tests for a HAL driver.

    Args:
        name: Base name for the test suite
        driver_name: Driver name for test registration
        driver_registration_hdr: Header file for driver registration
        driver_registration_fn: Function name for driver registration
        deps: Additional dependencies
        variant_suffix: Optional suffix for test names
        compiler_target_backend: Backend for executable compilation
        compiler_target_device: Device target for compilation
        compiler_flags: Additional compiler flags
        executable_format: Format string for executables
        included_tests: List of tests to include (defaults to all)
        excluded_tests: List of tests to exclude
        args: Additional test arguments
        tags: Test tags
        **kwargs: Additional arguments
    """

    tests_to_run = included_tests if included_tests else CTS_ALL_TESTS
    tests_to_run = [t for t in tests_to_run if t not in excluded_tests]

    # Generate executable data if needed
    executable_deps = []
    executables_testdata_hdr = ""

    if compiler_target_backend and executable_format:
        executable_deps = _generate_executable_testdata(
            compiler_target_backend, compiler_target_device, compiler_flags)
        executables_testdata_hdr = "{}_executables_c.h".format(compiler_target_backend)

    # Generate individual tests
    test_targets = []
    for test_name in tests_to_run:
        if test_name in CTS_EXECUTABLE_TESTS and not executable_deps:
            continue  # Skip executable tests if no backend configured

        test_target_name = _get_test_target_name(driver_name, variant_suffix, test_name)

        _generate_cts_test(
            name = test_target_name,
            test_name = test_name,
            driver_name = driver_name,
            driver_registration_hdr = driver_registration_hdr,
            driver_registration_fn = driver_registration_fn,
            executable_format = executable_format if test_name in CTS_EXECUTABLE_TESTS else "",
            executables_testdata_hdr = executables_testdata_hdr if test_name in CTS_EXECUTABLE_TESTS else "",
            compiler_target_backend = compiler_target_backend,
            compiler_target_device = compiler_target_device,
            deps = deps + executable_deps if test_name in CTS_EXECUTABLE_TESTS else deps,
            args = args,
            tags = tags + ["driver={}".format(driver_name)],
            **kwargs
        )
        test_targets.append(test_target_name)

    # Create test suite
    native.test_suite(
        name = name,
        tests = test_targets,
        tags = tags + ["driver={}".format(driver_name)],
    )

def _generate_executable_testdata(backend, device, flags):
    """Generate embedded executable data for tests that need it."""
    # Use backend name for both target and file naming
    # This matches the CMake implementation and avoids conflicts
    executables_testdata_name = "{}_executables".format(backend)

    translate_flags = ["--compile-mode=hal-executable"] + flags
    if device:
        translate_flags.append("--iree-hal-target-device={}".format(device))
    else:
        translate_flags.append("--iree-hal-target-backends={}".format(backend))

    # Generate bytecode modules for each source
    embed_data_sources = []
    for file_name in CTS_EXECUTABLE_SOURCES:
        # Use backend name for both target and module files to match CMake behavior
        module_name = "{}_{}".format(backend, file_name)
        module_file = "{}.bin".format(module_name)

        maybe(
            iree_bytecode_module,
            name = module_name,  # Use backend-based names to allow sharing
            src = "//runtime/src/iree/hal/cts/testdata:{}.mlir".format(file_name),
            flags = translate_flags,
            module_name = module_file,
            testonly = True,
        )
        embed_data_sources.append(module_file)

    # Embed the compiled modules as C data
    maybe(
        iree_c_embed_data,
        name = "{}_c".format(executables_testdata_name),
        srcs = embed_data_sources,
        c_file_output = "{}_c.c".format(executables_testdata_name),
        h_file_output = "{}_c.h".format(executables_testdata_name),
        identifier = "iree_cts_testdata_executables",
        strip_prefix = "{}_".format(backend),  # Use backend name for stripping
        flatten = True,
        testonly = True,
    )

    return [":{}".format("{}_c".format(executables_testdata_name))]

def _generate_cts_test(name, test_name, driver_name, driver_registration_hdr,
                      driver_registration_fn, executable_format, executables_testdata_hdr,
                      compiler_target_backend, compiler_target_device, deps, args, tags, **kwargs):
    """Generate a single CTS test."""

    # Generate the test source file
    source_name = "{}_source".format(name)
    cts_test_gen(
        name = source_name,
        template = "//runtime/src/iree/hal/cts:cts_test_template.cc.in",
        test_file_path = "runtime/src/iree/hal/cts/{}_test.h".format(test_name),
        driver_registration_hdr = driver_registration_hdr,
        driver_registration_fn = driver_registration_fn,
        driver_name = driver_name,
        executable_format = executable_format or "",
        executables_testdata_hdr = executables_testdata_hdr or "",
        target_backend = compiler_target_backend or "",
        target_device = compiler_target_device or "",
        testonly = True,
    )

    # Create the actual test
    test_deps = deps + [
        "//runtime/src/iree/hal/cts:{}_test_library".format(test_name),
        "//runtime/src/iree/hal/cts:cts_test_base",
        "//runtime/src/iree/base",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ]

    iree_runtime_cc_test(
        name = name,
        srcs = [":{}".format(source_name)],
        args = args,
        deps = test_deps,
        tags = tags,
        testonly = True,
        **kwargs
    )

def _get_test_target_name(driver_name, variant_suffix, test_name):
    """Get the target name for a test."""
    if variant_suffix:
        return "{}_{}_{}_test".format(driver_name, variant_suffix, test_name)
    else:
        return "{}_{}_test".format(driver_name, test_name)
