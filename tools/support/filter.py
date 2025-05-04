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


import random
import string

import panflute as pf  # type: ignore[import-not-found]


def action(elem: pf.Element, doc: pf.Doc) -> pf.Element | list[pf.Element] | None:
    if isinstance(elem, pf.CodeBlock):
        # Generate a random ID for the code block
        code_tag: str = "_" + "".join(random.choice(string.ascii_letters + string.digits) for i in range(12))
        elem.identifier = code_tag

        # Extract the language and filename from the Markdown code block identifier
        code_lang_and_filename: str = elem.classes and elem.classes[0] or ""

        # Only add the filename span if a filename is present
        filename_span: str = ""
        if ":" in code_lang_and_filename:
            lang, filename = code_lang_and_filename.split(":", 1)
            elem.classes[0] = lang  # Fix the language class
            filename_span = f"<span>{filename}</span>"

        return pf.Div(
            elem,
            pf.Div(pf.RawBlock(f"<button onclick=\"copyCode('{code_tag}')\">Copy</button><div></div>{filename_span}", "html"),
                   classes=["code-controls"]),
            classes=["code-container"],
        )

    return None


if __name__ == "__main__":
    pf.run_filter(action)
