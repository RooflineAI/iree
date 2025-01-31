load("@with_cfg.bzl", "with_cfg")

def copy_to_path(name, src, output_dir, file_name, *args, **kwargs):
    output_path = output_dir + "/" + file_name
    native.genrule(
        name = name,
        srcs = [src],
        outs = [output_path],
        cmd = "mkdir -p output_dir && cp $< $@",
  )



tracy_copy_to_path, _tracy_copy_to_path_internal = with_cfg(copy_to_path).set(Label("//runtime/src/iree/base/tracing:tracing_provider"), "tracy").build()
