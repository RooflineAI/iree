def _cts_test_gen_impl(ctx):
    """Implementation for generating CTS test source files."""
    output = ctx.actions.declare_file(ctx.attr.name + ".cc")

    substitutions = {
        "#cmakedefine IREE_CTS_TEST_FILE_PATH \"@IREE_CTS_TEST_FILE_PATH@\"": "#define IREE_CTS_TEST_FILE_PATH \"{}\"".format(ctx.attr.test_file_path),
        "#cmakedefine IREE_CTS_DRIVER_REGISTRATION_HDR \"@IREE_CTS_DRIVER_REGISTRATION_HDR@\"": "#define IREE_CTS_DRIVER_REGISTRATION_HDR \"{}\"".format(ctx.attr.driver_registration_hdr),
        "#cmakedefine IREE_CTS_DRIVER_REGISTRATION_FN @IREE_CTS_DRIVER_REGISTRATION_FN@": "#define IREE_CTS_DRIVER_REGISTRATION_FN {}".format(ctx.attr.driver_registration_fn),
        "#cmakedefine IREE_CTS_DRIVER_NAME \"@IREE_CTS_DRIVER_NAME@\"": "#define IREE_CTS_DRIVER_NAME \"{}\"".format(ctx.attr.driver_name),
        "#cmakedefine IREE_CTS_EXECUTABLE_FORMAT @IREE_CTS_EXECUTABLE_FORMAT@": "#define IREE_CTS_EXECUTABLE_FORMAT {}".format(ctx.attr.executable_format) if ctx.attr.executable_format else "/* #undef IREE_CTS_EXECUTABLE_FORMAT */",
        "#cmakedefine IREE_CTS_EXECUTABLES_TESTDATA_HDR \"@IREE_CTS_EXECUTABLES_TESTDATA_HDR@\"": "#define IREE_CTS_EXECUTABLES_TESTDATA_HDR \"{}\"".format(ctx.attr.executables_testdata_hdr) if ctx.attr.executables_testdata_hdr else "/* #undef IREE_CTS_EXECUTABLES_TESTDATA_HDR */",
        "#cmakedefine IREE_CTS_TARGET_BACKEND \"@IREE_CTS_TARGET_BACKEND@\"": "#define IREE_CTS_TARGET_BACKEND \"{}\"".format(ctx.attr.target_backend) if ctx.attr.target_backend else "/* #undef IREE_CTS_TARGET_BACKEND */",
        "#cmakedefine IREE_CTS_TARGET_DEVICE \"@IREE_CTS_TARGET_DEVICE@\"": "#define IREE_CTS_TARGET_DEVICE \"{}\"".format(ctx.attr.target_device) if ctx.attr.target_device else "/* #undef IREE_CTS_TARGET_DEVICE */",
    }

    ctx.actions.expand_template(
        template = ctx.file.template,
        output = output,
        substitutions = substitutions,
    )

    return [DefaultInfo(files = depset([output]))]

cts_test_gen = rule(
    implementation = _cts_test_gen_impl,
    attrs = {
        "template": attr.label(allow_single_file = True),
        "test_file_path": attr.string(),
        "driver_registration_hdr": attr.string(),
        "driver_registration_fn": attr.string(),
        "driver_name": attr.string(),
        "executable_format": attr.string(),
        "executables_testdata_hdr": attr.string(),
        "target_backend": attr.string(),
        "target_device": attr.string(),
    },
)
