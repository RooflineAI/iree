#!/usr/bin/env python3

import argparse
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template_in", required=True, help="Path to the input template file."
    )
    parser.add_argument(
        "--output_file", required=True, help="Path to the output C++ file."
    )
    parser.add_argument(
        "--sub",
        action="append",
        default=[],
        help='Substitution in format VAR_NAME=replacement_value_for_C. The replacement_value_for_C should already be formatted as it should appear in C code (e.g., "foo.h" or my_symbol).',
    )
    args = parser.parse_args()

    try:
        with open(args.template_in) as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Template file not found: {args.template_in}")
        return 1
    except Exception as e:
        print(f"Error reading template file {args.template_in}: {e}")
        return 1

    # Convert #cmakedefine to #define. Handles optional leading whitespace.
    content = re.sub(
        r"^[ \t]*#cmakedefine[ \t]+", "#define ", content, flags=re.MULTILINE
    )

    # Apply substitutions
    for sub_arg in args.sub:
        parts = sub_arg.split("=", 1)
        if len(parts) == 2:
            var_name, replacement_value = parts
            if not replacement_value:
                placeholder = f'#define {var_name} ["]?@{var_name}@["]?'
            else:
                placeholder = f"@{var_name}@"
            content = re.sub(
                placeholder, replacement_value, content, flags=re.MULTILINE
            )
            # content = content.replace(placeholder, replacement_value)
        else:
            print(f"Warning: Malformed substitution, skipping: {sub_arg}")

    try:
        with open(args.output_file, "w") as f:
            f.write(content)
    except Exception as e:
        print(f"Error writing output file {args.output_file}: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
