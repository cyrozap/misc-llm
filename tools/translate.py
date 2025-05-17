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
import sys
import time

import openai


MODELS: list[str] = [
    "qwen2.5-coder:32b-instruct-q4_K_M-17k",
    "granite3.3:8b-128k",
    "granite3.1-dense:8b-instruct-q4_K_M-128k",
]


def find_model_by_prefix(prefix: str) -> str:
    for model in MODELS:
        if model.startswith(prefix):
            return model
    raise ValueError(f"Model with prefix \"{prefix}\" not found. Available models are: {", ".join(MODELS)}")

def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Translate text to a specified language.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-k", "--api-key", type=str,
                        help="Your OpenAI API key (can also be set via the OPENAI_API_KEY environment variable)")
    parser.add_argument("-b", "--base-url", type=str,
                        help="Your OpenAI API base URL (can also be set via the OPENAI_BASE_URL environment variable)")
    parser.add_argument("-m", "--model", type=find_model_by_prefix, default=MODELS[0],
                        help="Model to use. You can provide a prefix match for any of the following models (default: {}):\n\n{} ".format(
                            MODELS[0], "\n".join(["- {}".format(model) for model in MODELS])))
    parser.add_argument("-l", "--language", type=str, default="English",
                        help="Language to translate text into (default: English)")

    return parser.parse_args()

def main() -> int:
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

    input_data: str = sys.stdin.read().strip()

    system_message: str = f"Translate the provided text to {args.language}. YOU MUST ONLY OUTPUT THE TRANSLATED TEXT!"

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": input_data,
        },
    ]

    start_time: float = time.time()

    response: openai.Stream[openai.types.chat.ChatCompletionChunk] = client.chat.completions.create(
        model=args.model,
        messages=messages,  # type: ignore[arg-type]
        stream=True,
        stream_options={"include_usage": True},  # type: ignore[call-overload]
    )

    usage: openai.types.chat.chat_completion.CompletionUsage | None = None
    for chunk in response:
        if not chunk:
            break
        usage = chunk.usage
        choices: list[openai.types.chat.chat_completion_chunk.Choice] = chunk.choices
        if choices:
            chunk_content: str | None = choices[0].delta.content
            if chunk_content is not None:
                print(chunk_content, end="", flush=True)
    print()

    end_time: float = time.time()

    if usage:
        total_time: float = end_time - start_time

        token_usage_message: str = "\n".join([
            "",
            "Tokens used: {}".format(usage.total_tokens),  # type: ignore[union-attr]
            "Prompt tokens: {}".format(usage.prompt_tokens),  # type: ignore[union-attr]
            "Completion tokens: {}".format(usage.completion_tokens),  # type: ignore[union-attr]
            "Time taken: {:.6f} seconds".format(total_time),
            "Tokens per second: {:.2f} tokens/s".format(usage.completion_tokens / total_time),  # type: ignore[union-attr]
        ])

        print(token_usage_message, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
