#!/usr/bin/env python3
"""Generate the JIT preamble string for the KTT C++ backend.

Reads Source/ComputeEngine/Cpp/CppKernelDefines.h (the single source of truth for the
ktt::ScalarType union) and writes it into Source/ComputeEngine/Cpp/CppKernelDefines.generated.h
as a raw string constant that CppCompiler injects into every JIT-compiled kernel.

The generated file is a build artifact and must never be edited by hand. The script rewrites the
output only when its content would change, so repeated builds do not touch the file needlessly.
"""

import sys
from pathlib import Path

RAW_STRING_DELIMITER = "kttdefines"


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <CppKernelDefines.h> <output header>", file=sys.stderr)
        return 1

    headerPath = Path(sys.argv[1])
    outputPath = Path(sys.argv[2])

    text = headerPath.read_text(encoding="utf-8")
    if f"){RAW_STRING_DELIMITER}\"" in text:
        print(f"Error: {headerPath} contains the raw string delimiter {RAW_STRING_DELIMITER}", file=sys.stderr)
        return 1

    content = (
        "// Generated from " + headerPath.name + " by Scripts/generate_cpp_kernel_defines.py."
        " Do not edit; edit the source header instead.\n"
        "#pragma once\n"
        "\n"
        "namespace ktt\n"
        "{\n"
        "\n"
        "// Full text of " + headerPath.name + ", injected as a preamble into JIT-compiled C++ kernels.\n"
        "inline constexpr char kttCppKernelDefinesSource[] = R\"" + RAW_STRING_DELIMITER + "("
        + text + ")" + RAW_STRING_DELIMITER + "\";\n"
        "\n"
        "} // namespace ktt\n"
    )

    if outputPath.exists() and outputPath.read_text(encoding="utf-8") == content:
        return 0

    outputPath.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
