#!/bin/bash
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


set -euo pipefail


if [ "$#" -eq 0 ]; then
	echo "Usage: $0 <file1> [<file2> ... <fileN>]"
	exit 1
fi

for file in "$@"; do
	model_name=$(grep -m 1 '^FROM' "${file}" | awk '{print $2}')
	if [ -z "${model_name}" ]; then
		echo "No valid FROM line found in file: ${file}"
		continue
	fi

	echo "Pulling model: ${model_name}"
	ollama pull "${model_name}"
done
