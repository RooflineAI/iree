def _workspace_root_impl(ctx):
    """Dynamically determine the workspace root from the current context.
    The path is made available as a `WORKSPACE_ROOT` environmment variable and
    may for instance be consumed in the `toolchains` attributes for `cc_library`
    and `genrule` targets.
    """
    workspace_root = ctx.label.workspace_root if ctx.label.workspace_root else "."
    return [
        platform_common.TemplateVariableInfo({
            "WORKSPACE_ROOT": workspace_root,
        }),
    ]

workspace_root = rule(
    implementation = _workspace_root_impl,
    attrs = {},
)
