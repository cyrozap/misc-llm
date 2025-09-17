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
from enum import Enum

import openai


MODELS: list[str] = [
    "qwen3:30b-a3b-instruct-2507-q4_K_M-117k",
]


class ThinkingState(Enum):
    PRE_THINKING = 0
    THINKING = 1
    END_THINKING = 2
    DONE_THINKING = 3


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

    if args.model.startswith("qwen3:") and not args.model.startswith("qwen3:30b-a3b-instruct-2507"):
        system_message += " /no_think"

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

    response_start: float | None = None
    think_start: float | None = None
    think_end: float | None = None

    thinking_state: ThinkingState = ThinkingState.PRE_THINKING

    usage: openai.types.chat.chat_completion.CompletionUsage | None = None
    for chunk in response:
        if not chunk:
            break
        if thinking_state == ThinkingState.END_THINKING:
            thinking_state = ThinkingState.DONE_THINKING
        usage = chunk.usage
        choices: list[openai.types.chat.chat_completion_chunk.Choice] = chunk.choices
        if choices:
            chunk_content: str | None = choices[0].delta.content
            if chunk_content is not None:
                if response_start is None:
                    response_start = time.time()

                if "<think>" in chunk_content and think_start is None:
                    think_start = time.time()
                    if thinking_state == ThinkingState.PRE_THINKING:
                        thinking_state = ThinkingState.THINKING

                if "</think>" in chunk_content and think_end is None:
                    think_end = time.time()
                    if thinking_state == ThinkingState.THINKING:
                        thinking_state = ThinkingState.END_THINKING

                if thinking_state in (ThinkingState.PRE_THINKING, ThinkingState.DONE_THINKING):
                    print(chunk_content, end="", flush=True)
    print()

    end_time: float = time.time()

    if response_start is None:
        response_start = end_time

    thinking_time: float = 0
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
            "Total time taken: {:.6f} seconds".format(total_time),
            "Time spent thinking: {:.6f} seconds".format(thinking_time),
            "Prompt processing speed: {:.2f} tokens/s".format(usage.prompt_tokens / pp_time),  # type: ignore[union-attr]
            "Generation speed: {:.2f} tokens/s".format(usage.completion_tokens / gen_time),  # type: ignore[union-attr]
        ])

        print(token_usage_message, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
