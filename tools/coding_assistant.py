#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD

# Copyright (C) 2025 by Forest Crossman <cyrozap@gmail.com>
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
import tempfile
import time
from pathlib import Path

import openai
from bs4 import BeautifulSoup
from bs4.element import PageElement, Tag


MODELS: list[str] = [
    "qwen2.5-coder:32b-instruct-q4_K_M-17k",
    "devstral:24b-small-2505-q4_K_M-61k",
    "qwen3:14b-q4_K_M-95k",
    "qwen3:30b-a3b-q4_K_M-46k",
    "qwen3:32b-q4_K_M-12k",
    "deepseek-r1:32b-qwen-distill-q4_K_M-17k",
    "qwq:32b-q4_K_M-17k",
    "granite3.3:8b-128k",
    "granite3.1-dense:8b-instruct-q4_K_M-128k",

    # 48 GB VRAM only
    "qwen2.5-coder:32b-instruct-q4_K_M-110k",
    "deepseek-r1:32b-qwen-distill-q4_K_M-110k",
]

LANGUAGE_EXPERTISE: list[str] = [
    "Python",
    "Java",
    "Rust",
    "C",
    "Verilog",
]

SKILLS: list[str] = [
    "software reverse engineering",
    "hardware reverse engineering",
]


def find_model_by_prefix(prefix: str) -> str:
    for model in MODELS:
        if model.startswith(prefix):
            return model
    raise ValueError(f"Model with prefix \"{prefix}\" not found. Available models are: {", ".join(MODELS)}")

def format_duration(seconds: float | int) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{minutes}m{remaining_seconds}s" if remaining_seconds > 0 else f"{minutes}m"
    else:
        hours, remaining = divmod(seconds, 3600)
        minutes, seconds = divmod(remaining, 60)
        parts: list[str] = [
            f"{hours}h" if hours > 0 else "",
            f"{minutes}m" if minutes > 0 or not hours else "",
            f"{seconds}s" if seconds > 0 or not minutes and not hours else "",
        ]
        return "".join(parts)

def join_with_and(strs: list[str]) -> str:
    if len(strs) == 0:
        return ""
    elif len(strs) == 1:
        return strs[0]
    elif len(strs) == 2:
        return f"{strs[0]} and {strs[1]}"
    else:
        all_but_last: str = ", ".join(strs[:-1])
        last_string: str = strs[-1]
        return f"{all_but_last}, and {last_string}"

def replace_think_tag(html_data: bytes, thinking_time: float | None) -> bytes:
    soup: BeautifulSoup = BeautifulSoup(html_data.decode("utf-8"), "html.parser")

    think: PageElement | None = soup.find("think")
    if think is None or not isinstance(think, Tag):
        return html_data

    details: Tag = soup.new_tag("details")

    summary: Tag = soup.new_tag("summary")
    summary.string = "💡 Thought Process"
    if thinking_time is not None:
        summary.string += " (thought for {})".format(format_duration(thinking_time))
    details.append(summary)

    div: Tag = soup.new_tag("div")
    div.append(BeautifulSoup(think.decode_contents(), "html.parser"))
    details.append(div)

    think.replace_with(details)

    return str(soup).encode("utf-8")

def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Coding Assistant.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-k", "--api-key", type=str,
                        help="Your OpenAI API key (can also be set via the OPENAI_API_KEY environment variable)")
    parser.add_argument("-b", "--base-url", type=str,
                        help="Your OpenAI API base URL (can also be set via the OPENAI_BASE_URL environment variable)")
    parser.add_argument("-n", "--no-browser", action="store_true",
                        help="When enabled, don't open in a browser window.")
    parser.add_argument("-m", "--model", type=find_model_by_prefix, default=MODELS[0],
                        help="Model to use. You can provide a prefix match for any of the following models (default: {}):\n\n{} ".format(
                            MODELS[0], "\n".join(["- {}".format(model) for model in MODELS])))
    parser.add_argument("user_prompt", help="The user's question or prompt", type=str)
    parser.add_argument("filenames", nargs="*", help="list of filenames to provide as context", type=str)

    return parser.parse_args()

def main() -> None:
    args: argparse.Namespace = parse_args()

    # Read from environment variables if not provided via command-line arguments
    api_key: str = args.api_key or os.getenv("OPENAI_API_KEY") or "dummy"
    base_url: str | None = args.base_url or os.getenv("OPENAI_BASE_URL")

    if not base_url:
        raise ValueError("Missing OpenAI API base URL")

    client: openai.OpenAI = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    system_message_parts: list[str] = [
        "You are Koda, an AI coding assistant.",
        f"You are an expert in {join_with_and(LANGUAGE_EXPERTISE)}.",
        f"You are also an expert at {join_with_and(SKILLS)}.",
        "Do NOT prefix your responses with any words like \"Certainly!\", \"Sure!\", or similar phrases.",
        "If your answer contains fenced code blocks in Markdown, include the relevant full file path in the code block tag using this structure: ```$LANGUAGE:$FILEPATH```",
        "For example, for a Python file \"program.py\", the structure should be: ```python:program.py```",
        "For executable terminal commands, enclose each command in an individual ```bash``` language fenced code block without any comments or newlines inside.",
    ]

    system_message: str = "\n".join(system_message_parts)

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": system_message,
        },
    ]

    if args.filenames:
        for filename in args.filenames:
            with open(filename, "r") as f:
                file_content: str = "Content for \"{}\":\n\n```\n{}\n```\n".format(filename, f.read().strip())
                # file_content: str = "Content for \"{}\": \"{}\"".format(filename, f.read())

                messages.append({"role": "user", "content": file_content})
                messages.append({"role": "assistant", "content": "Ok."})

    final_prompt_parts: list[str] = [
        "Answer positively without apologizing.",
    ]

    if args.filenames:
        final_prompt_parts.append("You have access to the provided codebase context.")

    final_prompt_parts.append("Question:")

    if args.filenames:
        for filename in args.filenames:
            final_prompt_parts.append("`{}`".format(filename))

    final_prompt_parts.append(args.user_prompt)

    final_prompt: str = " ".join(final_prompt_parts)

    messages.append({"role": "user", "content": final_prompt})

    start_time: float = time.time()

    response: openai.Stream[openai.types.chat.ChatCompletionChunk] = client.chat.completions.create(
        model=args.model,
        messages=messages,  # type: ignore[arg-type]
        stream=True,
        stream_options={"include_usage": True},  # type: ignore[call-overload]
    )

    print(f"Model: {args.model}", file=sys.stderr)

    response_start: float | None = None
    think_start: float | None = None
    think_end: float | None = None

    usage: openai.types.chat.chat_completion.CompletionUsage | None = None
    collected_response: str = ""
    for chunk in response:
        if not chunk:
            break
        usage = chunk.usage
        choices: list[openai.types.chat.chat_completion_chunk.Choice] = chunk.choices
        if choices:
            chunk_content: str | None = choices[0].delta.content
            if chunk_content is not None:
                if response_start is None:
                    response_start = time.time()
                if "<think>" in chunk_content and think_start is None:
                    think_start = time.time()
                if "</think>" in chunk_content and think_end is None:
                    think_end = time.time()
                print(chunk_content, end="", flush=True)
                collected_response += chunk_content
    print()

    end_time: float = time.time()

    if response_start is None:
        response_start = end_time

    thinking_time: float | None = None
    if think_start is not None and think_end is not None:
        thinking_time = think_end - think_start

    if usage:
        pp_time: float = response_start - start_time
        gen_time: float = end_time - response_start
        total_time: float = end_time - start_time

        token_usage_message: str = "\n".join([
            "",
            "Tokens used: {}".format(usage.total_tokens),  # type: ignore[union-attr]
            "Prompt tokens: {}".format(usage.prompt_tokens),  # type: ignore[union-attr]
            "Completion tokens: {}".format(usage.completion_tokens),  # type: ignore[union-attr]
            "Time taken: {:.6f} seconds".format(total_time),
            "Prompt processing speed: {:.2f} tokens/s".format(usage.prompt_tokens / pp_time),  # type: ignore[union-attr]
            "Generation speed: {:.2f} tokens/s".format(usage.completion_tokens / gen_time),  # type: ignore[union-attr]
        ])

        print(token_usage_message, file=sys.stderr)

    if args.no_browser:
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md") as markdown_file:
        markdown_paragraphs: list[str] = [
            "**Model:** `{}`".format(args.model),
            "## Prompt",
            args.user_prompt,
        ]

        if args.filenames:
            markdown_paragraphs.append("**Context:**")
            markdown_paragraphs.append("\n".join(["- `{}`".format(filename) for filename in args.filenames]))

        markdown_paragraphs.append("## Response")
        markdown_paragraphs.append(collected_response)

        markdown_file.write("\n\n".join(markdown_paragraphs))
        markdown_file.flush()

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".html") as html_file:
            script_directory: Path = Path(__file__).resolve().parent
            css_path: Path = script_directory / "resources" / "gh-pandoc.css"
            copy_script_html_path: Path = script_directory / "resources" / "copy.html"
            filter_path: Path = script_directory / "support" / "filter_wrapper.sh"

            pandoc_args: list[str] = [
                "pandoc",
                "--embed-resources",
                "--standalone",
                "--css", str(css_path),
            ]

            if (script_directory / "support" / ".env" / "bin" / "python3").is_file():
                pandoc_args.extend([
                    "--include-in-header", str(copy_script_html_path),
                    "--filter", str(filter_path),
                ])

            pandoc_args.extend([
                "--highlight-style", "kate",
                "--metadata", "title=Coding Assistant",
                "-f", "gfm",
                "-t", "html",
                markdown_file.name,
            ])

            output: subprocess.CompletedProcess = subprocess.run(
                pandoc_args, stdout=subprocess.PIPE)

            if output.returncode == 0:
                replaced: bytes = replace_think_tag(output.stdout, thinking_time)
                html_file.write(replaced)
                html_file.flush()

                subprocess.run(["chromium", "--incognito", html_file.name])

                time.sleep(3)


if __name__ == "__main__":
    main()
