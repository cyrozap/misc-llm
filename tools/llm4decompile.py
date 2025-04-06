#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD

# Copyright (C) 2024-2025 by Forest Crossman <cyrozap@gmail.com>
#
# Permission to use, copy, modify, and/or distribute this software for
# any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE
# AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
# PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
# TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.


import argparse
import os
import subprocess
import sys

import requests  # type: ignore


def format_with_clang_format(code: str, style: str = "") -> str:
    """Format the given code using clang-format."""
    try:
        command: list[str] = ["clang-format"]
        if style:
            command.extend(["--style", style])
        result: subprocess.CompletedProcess = subprocess.run(
            command,
            input=code,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running clang-format: {}".format(e), file=sys.stderr)
        return code  # Return the original code if formatting fails

def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Use an LLM to enhance code decompiled by Ghidra.")
    parser.add_argument("-r", "--raw", action="store_true", default=False, help="Print the output without formatting")
    parser.add_argument("--style", type=str, default="", help="Specify a style for clang-format")
    args: argparse.Namespace = parser.parse_args()

    input_data: str = sys.stdin.read().strip()

    payload = {
        "model": "llm4decompile:22b-v2-q6_K",
        "options": {
            "num_ctx": 26*1024,
        },
        #"model": "llm4decompile:9b-v2-q4_K_M",
        #"options": {
        #    "num_ctx": 128*1024,
        #},
        "stream": False,
        "prompt": input_data,
    }

    ollama_host: str = os.getenv("OLLAMA_HOST", "localhost:11434")
    if not (ollama_host.startswith("http://") or ollama_host.startswith("https://")):
        ollama_host = "http://" + ollama_host
    url: str = "{}/api/generate".format(ollama_host)

    response: requests.Response = requests.post(url, json=payload)
    response.raise_for_status()

    llm_code: str = response.json()["response"].strip()

    if not llm_code:
        print("Error: LLM failed to enhance code.", file=sys.stderr)
        return 1

    if not args.raw:
        # Format the output using clang-format
        llm_code = format_with_clang_format(llm_code, style=args.style).strip()

    print(llm_code)

    return 0


if __name__ == "__main__":
    sys.exit(main())
