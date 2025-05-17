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
import sys
import json
from typing import NamedTuple

import openai


MODELS: list[str] = [
    "qwen2.5-coder:32b-instruct-q4_K_M-17k",
    # "granite3.1-dense:8b-instruct-q4_K_M-128k",
]


class Error(NamedTuple):
    message: str


def is_text_file(filepath: str) -> bool:
    try:
        with open(filepath, "tr", encoding="utf-8"):
            pass
        return True
    except UnicodeDecodeError:
        return False

def find_model_by_prefix(prefix: str) -> str:
    for model in MODELS:
        if model.startswith(prefix):
            return model
    raise ValueError(f"Model with prefix \"{prefix}\" not found. Available models are: {", ".join(MODELS)}")

def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Check files against a specified prompt using an LLM.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-k", "--api-key", type=str,
                        help="Your OpenAI API key (can also be set via the OPENAI_API_KEY environment variable)")
    parser.add_argument("-b", "--base-url", type=str,
                        help="Your OpenAI API base URL (can also be set via the OPENAI_BASE_URL environment variable)")
    parser.add_argument("-m", "--model", type=find_model_by_prefix, default=MODELS[0],
                        help="Model to use. You can provide a prefix match for any of the following models (default: {}):\n\n{} ".format(
                            MODELS[0], "\n".join(["- {}".format(model) for model in MODELS])))
    parser.add_argument("prompt", type=str,
                        help="The prompt to check against each file's content")

    return parser.parse_args()

def process_file(filepath: str, client: openai.OpenAI, system_message: str, args: argparse.Namespace) -> str | Error | None:
    if not is_text_file(filepath):
        return None

    contents: str = open(filepath, "r", encoding="utf-8").read().strip()

    user_message: str = f"Content for \"{filepath}\":\n\n```\n{contents}\n```\n"

    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": user_message,
        },
        {
            "role": "assistant",
            "content": "Ok.",
        },
        {
            "role": "system",
            "content": system_message,
        },
    ]

    response: openai.types.chat.ChatCompletion | openai.Stream[openai.types.chat.ChatCompletionChunk] = client.chat.completions.create(
        model=args.model,
        messages=messages,  # type: ignore[arg-type]
        stream=True,
    )

    collected_response: str = ""
    if isinstance(response, openai.types.chat.ChatCompletion):
        completion_choices: list[openai.types.chat.chat_completion.Choice] = response.choices
        if completion_choices:
            completion_message: openai.types.chat.chat_completion_message.ChatCompletionMessage = completion_choices[0].message
            completion_content: str | None = completion_message.content
            if completion_content is not None:
                collected_response = completion_content
    else:
        for chunk in response:
            if not chunk:
                break
            chunk_choices: list[openai.types.chat.chat_completion_chunk.Choice] = chunk.choices
            if chunk_choices:
                chunk_content: str | None = chunk_choices[0].delta.content
                if chunk_content is not None:
                    collected_response += chunk_content

    try:
        json_response: dict = json.loads(collected_response.strip())
        if json_response.get("match", False):
            return filepath
    except json.JSONDecodeError:
        return Error(f"Failed to decode JSON from LLM for file \"{filepath}\"")

    return None

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

    system_message_parts: list[str] = [
        f"Given the file name and file contents provided by the user, determine whether it matches the following criteria: {args.prompt}",
        "Your response should be a JSON object containing a single boolean key `match`.",
        "The value of this key should be `true` if the criteria was met, otherwise it should be `false`.",
        "DO NOT OUTPUT ANY TEXT OTHER THAN THE JSON OBJECT!",
        "DO NOT PLACE THE JSON OBJECT INSIDE A CODE BLOCK!",
        "ALL JSON MUST BE PROPERLY FORMATTED WITH NO EMBEDDED COMMENTS!",
    ]

    system_message: str = "\n".join(system_message_parts)

    while True:
        filepath: str = sys.stdin.readline().strip()
        if not filepath:
            break

        result: str | Error | None = process_file(filepath, client, system_message, args)
        if isinstance(result, Error):
            print(result.message, file=sys.stderr)
        elif result:
            print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
