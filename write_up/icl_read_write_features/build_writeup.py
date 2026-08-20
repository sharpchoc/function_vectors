#!/usr/bin/env python3
"""Build the self-contained write-up HTML from writeup_template.html.

Every `{{IMG:<relative path>}}` placeholder is replaced by the PNG at
results/69_task_run/<relative path>, inlined as a base64 data URI, so the output
icl_read_write_features.html opens offline in any browser with no dependencies
(the Google Fonts link degrades to the system fallback stack when offline).

Usage (from anywhere):  python3 write_up/icl_read_write_features/build_writeup.py
"""
import base64
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "results" / "69_task_run"
template = (HERE / "writeup_template.html").read_text()

missing = []


def repl(m):
    rel = m.group(1)
    p = ROOT / rel
    if not p.exists():
        missing.append(rel)
        return "MISSING:" + rel
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


body = re.sub(r"\{\{IMG:([^}]+)\}\}", repl, template)
if missing:
    print("MISSING FILES:")
    for x in missing:
        print(" ", x)
    sys.exit(1)

# The template is a page fragment (the Claude artifact viewer supplies the document
# skeleton); wrap it so the file is a complete standalone document.
html = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
out = HERE / "icl_read_write_features.html"
out.write_text(html)
print(f"OK: {out} ({out.stat().st_size/1e6:.2f} MB, "
      f"{html.count('data:image/png;base64,')} images embedded)")
