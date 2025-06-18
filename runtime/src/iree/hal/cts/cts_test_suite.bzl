"""Defines a test suite for IREE HAL CTS tests."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

_ALL_CTS_TESTS = [
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

_EXECUTABLE_CTS_TESTS = [
    "command_buffer_dispatch",
    "command_buffer_dispatch_constants",
    "executable_cache",
]

_ALL_CTS_EXECUTABLE_SOURCES = [
    "command_buffer_dispatch_test",
    "command_buffer_dispatch_constants_test",
    "executable_cache_test",
]

def iree_hal_cts_test_suite(
        name,
        driver_name,
        driver_registration_hdr,
        driver_registration_fn,
        executable_format,
        variant_suffix = "",
        compiler_target_backend = "",
        compiler_target_device = "",
        args = [],
        deps = [],
        included_tests = None,
        excluded_tests = [],
        labels = []):
    """Defines a CTS test suite."""

    generated_bin_files = []
    executable_embed_genrule_target = None
    embedded_c_src_file = None

    if compiler_target_backend:  # Use the derived  value
        for mlir_source_name in _ALL_CTS_EXECUTABLE_SOURCES:
            output_bin_name = "%s_%s.bin" % (compiler_target_backend, mlir_source_name)
            compile_gen_rule_name = "%s_%s_module" % (compiler_target_backend, mlir_source_name)
            generated_bin_files.append(":" + compile_gen_rule_name)

            additional_compiler_flags = ""
            if compiler_target_device:  # Use the derived  value
                additional_compiler_flags = "--iree-hal-target-device=%s" % compiler_target_device

            native.genrule(
                name = compile_gen_rule_name,
                srcs = ["//runtime/src/iree/hal/cts/testdata:%s.mlir" % mlir_source_name],
                outs = [output_bin_name],
                cmd = "$(location //tools:iree-compile) --compile-mode=hal-executable --iree-hal-target-backends=%s %s -o $@ $<" % (
                    compiler_target_backend,
                    additional_compiler_flags,
                ),
                tools = ["//tools:iree-compile"],
            )

        embed_gen_name = "%s_executables_c" % compiler_target_backend
        embedded_c_src_file = embed_gen_name + ".c"
        embed_h_file = embed_gen_name + ".h"

        executable_embed_genrule_target = ":" + embed_gen_name

        srcs_for_embed = " ".join(["$(location %s)" % bin_target for bin_target in generated_bin_files])

        native.genrule(
            name = embed_gen_name,
            srcs = generated_bin_files,
            outs = [
                embedded_c_src_file,
                embed_h_file,
            ],
            cmd = "$(location //build_tools/embed_data:iree-c-embed-data) --identifier=iree_cts_testdata_executables --strip_prefix=%s_ --flatten --output_impl=$(location %s) --output_header=$(location %s) %s" % (
                compiler_target_backend,
                embedded_c_src_file,
                embed_h_file,
                srcs_for_embed,
            ),
            tools = ["//build_tools/embed_data:iree-c-embed-data"],
        )
        cc_library(
            name = embed_gen_name + "_cc",
            srcs = [embedded_c_src_file],
            hdrs = [embed_h_file],
            deps = [],  # or other libs if needed
        )

    tests_to_generate = []
    if included_tests != None:
        tests_to_generate = [
            test
            for test in included_tests
            if test not in excluded_tests
        ]
    else:
        tests_to_generate = [
            test
            for test in _ALL_CTS_TESTS
            if test not in excluded_tests
        ]

    template_label = "//runtime/src/iree/hal/cts:cts_test_template.cc.in"

    for test_name in tests_to_generate:
        cc_test_target_name_base = "%s_%s" % (driver_name, test_name)
        if variant_suffix:
            cc_test_target_name_base = "%s_%s_%s" % (driver_name, variant_suffix, test_name)

        generated_cc_src_file = cc_test_target_name_base + "_test.cc"
        cc_test_target_name = cc_test_target_name_base + "_test"

        # Logic for executable_testdata_hdr_val based on new prompt
        executable_testdata_hdr_val = ""  # Default to empty string
        if compiler_target_backend and test_name in _EXECUTABLE_CTS_TESTS:
            executable_testdata_hdr_val = "%s_executables_c.h" % compiler_target_backend

        genrule_srcs_for_test_cc = [template_label]
        genrule_tools_for_test_cc = [template_label]
        if test_name in _EXECUTABLE_CTS_TESTS and executable_embed_genrule_target:
            genrule_tools_for_test_cc.append(executable_embed_genrule_target)

        # Values for sed substitution (plain Python strings, sed will add quotes)
        # Note: driver_registration_fn is a C symbol, so it's not quoted in the sed replacement
        sed_cmd_parts = [
            "--sub=\"IREE_CTS_TEST_FILE_PATH=runtime/src/iree/hal/cts/%s_test.h\"" % test_name,
            "--sub=\"IREE_CTS_DRIVER_REGISTRATION_HDR=%s\"" % driver_registration_hdr,
            "--sub=\"IREE_CTS_DRIVER_REGISTRATION_FN=%s\"" % driver_registration_fn,
            "--sub=\"IREE_CTS_DRIVER_NAME=%s\"" % driver_name,
            "--sub=\"IREE_CTS_EXECUTABLE_FORMAT=%s\"" % ("\\\"%s\\\"" % executable_format if test_name in _EXECUTABLE_CTS_TESTS else ""),
            "--sub=\"IREE_CTS_EXECUTABLES_TESTDATA_HDR=%s\"" % executable_testdata_hdr_val,  # executable_testdata_hdr_val is already just filename or empty
            "--sub=\"IREE_CTS_TARGET_BACKEND=%s\"" % (compiler_target_backend if test_name in _EXECUTABLE_CTS_TESTS else ""),
            "--sub=\"IREE_CTS_TARGET_DEVICE=%s\"" % (compiler_target_device if test_name in _EXECUTABLE_CTS_TESTS else ""),
        ]
        sed_cmd = (" ".join(sed_cmd_parts))
        print(test_name)
        print(sed_cmd)  # For debugging purposes, can be removed later

        native.genrule(
            name = generated_cc_src_file + "_gen",
            srcs = genrule_srcs_for_test_cc,
            outs = [generated_cc_src_file],
            cmd = "$(location //runtime/src/iree/hal/cts:process_cts_template) --template_in=$(SRCS) --output_file=$@ %s" % (sed_cmd),
            tools = ["//runtime/src/iree/hal/cts:process_cts_template"],
        )

        test_srcs = [":" + generated_cc_src_file + "_gen"]
        test_deps = [
            "//runtime/src/iree/hal/cts:%s_test_library" % test_name,
            "//runtime/src/iree/hal/cts:cts_test_base",
            "//runtime/src/iree/base",
            "//runtime/src/iree/hal",
            "@iree//runtime/src/iree/testing:gtest",
            "@iree//runtime/src/iree/testing:gtest_main",
        ] + deps

        if test_name in _EXECUTABLE_CTS_TESTS and compiler_target_backend and embedded_c_src_file and executable_embed_genrule_target:
            # alternative A
            # test_srcs.append(embedded_c_src_file)  # Not needed, already included
            # test_srcs.append(embed_h_file)  # Not needed, already included

            # alternative B
            if executable_embed_genrule_target not in test_deps:
                test_deps.append(":" + embed_gen_name + "_cc")

        test_tags = list(labels)
        test_tags.append("driver=%s" % driver_name)
        if variant_suffix:
            test_tags.append("variant=%s" % variant_suffix)
        cc_test(
            name = cc_test_target_name,
            srcs = test_srcs,
            deps = test_deps,
            args = args,
            tags = test_tags,
        )
