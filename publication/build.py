#!/usr/bin/env python3
"""Build the publication PDF from the locked manuscript.

Run from this directory:
    PYTHONDONTWRITEBYTECODE=1 python3 -B build.py

Reads   ../CEG-JOURNAL-MANUSCRIPT-V3.2.md   (never modified)
Writes  ./CEG-JOURNAL-MANUSCRIPT-V3.2.typ   (generated Typst source)
        ./CEG-JOURNAL-MANUSCRIPT-V3.2.pdf
        ../CEG-JOURNAL-MANUSCRIPT-V3.2.pdf  (copy)

Requires the `pandoc` and `typst` executables on PATH and ./figures/*.pdf
from make_figures.py. Only the Python standard library is used here.

The manuscript text is converted mechanically by pandoc. This script changes
presentation only: it replaces the two Mermaid blocks and adds Figure 3 as
vector images, wraps the title block, abstract, captions and reference list in
layout containers, and gives tables booktabs rules and column widths.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "CEG-JOURNAL-MANUSCRIPT-V3.2"
SOURCE = HERE.parent / f"{STEM}.md"
FIGURES = {
    1: "figures/fig1_context_poset_hasse.pdf",
    2: "figures/fig2_two_regime_workflow.pdf",
    3: "figures/fig3_empirical_summary.pdf",
}
EXPECT = {"figures": 3, "mermaid": 2, "tables": 14}


def fail(msg: str) -> None:
    sys.exit(f"build.py: {msg}")


def raw(code: str) -> list[str]:
    return ["", "```{=typst}", code, "```", ""]


def visible(cell: str) -> str:
    """Approximate the printed text of a Markdown table cell."""
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", cell)
    s = re.sub(r"\\[a-zA-Z]+", "x", s)
    return re.sub(r"[`*$\\{}]", "", s).strip()


def column_spec(rows: list[list[str]]) -> str:
    """Short columns size to their content; long ones share the remaining width."""
    ncol = len(rows[0])
    spec = []
    for j in range(ncol):
        lens = [len(visible(r[j])) for r in rows if j < len(r)]
        longest = max(lens)
        if longest <= 14:
            spec.append("auto")
        else:
            body = lens[1:] or lens
            weight = max(1.0, min(4.0, (sum(body) / len(body)) / 11.0))
            spec.append(f"{weight:.2f}fr")
    return "(" + ", ".join(spec) + ",)"


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def transform(md: str) -> tuple[list[str], list[str], str]:
    lines = md.split("\n")
    if not lines[0].startswith("# "):
        fail("manuscript does not start with a level-1 title")
    title = lines[0][2:].strip()
    out: list[str] = raw("#frontmatter[")
    colspecs: list[str] = []
    counts = {"figures": 0, "mermaid": 0, "tables": 0}
    rules = 0
    in_refs = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "---":
            rules += 1
            if rules == 1:
                out += raw("]") + raw("#abstractblock[")
            elif rules == 2:
                out += raw("]")
            i += 1
            continue
        if line.startswith("```mermaid"):
            counts["mermaid"] += 1
            i += 1
            while not lines[i].startswith("```"):
                i += 1
            i += 1
            while not lines[i].strip():
                i += 1
            m = re.match(r"\*Figure (\d+)\. ", lines[i])
            if not m:
                fail("a Mermaid block is not followed by its figure caption")
            line = lines[i]
        m = re.match(r"\*Figure (\d+)\. ", line)
        if m:
            n = int(m.group(1))
            counts["figures"] += 1
            if n not in FIGURES or n != counts["figures"]:
                fail(f"unexpected figure number {n}")
            out += raw(f'#figblock[#figimage("{FIGURES[n]}")') + [line] + raw("]")
            i += 1
            continue
        if line.startswith("**Table ") and line.rstrip().endswith("**"):
            counts["tables"] += 1
            out += raw("#tablecaption[") + [line] + raw("]")
            j = i + 1
            while not lines[j].strip():
                j += 1
            if not lines[j].startswith("|"):
                fail(f"no table follows the caption: {line[:40]}")
            k = j
            while k < len(lines) and lines[k].startswith("|"):
                k += 1
            rows = [split_row(l) for l in lines[j:k] if not re.match(r"^\|[\s:|-]+\|$", l)]
            colspecs.append(column_spec(rows))
            i += 1
            continue
        out.append(line)
        if line.strip() == "## References":
            in_refs = True
            out += raw("#references[")
        i += 1
    if not in_refs:
        fail("no References heading found")
    out += raw("]")
    if counts != EXPECT or rules < 2:
        fail(f"structure differs from expectation: {counts}, {rules} rules")
    return out, colspecs, title


def postprocess(typ: str, colspecs: list[str]) -> str:
    # \mathrm{word} arrives as upright(w o r d); join the letters into one word.
    def join(m: re.Match) -> str:
        return 'upright("' + m.group(1).replace(" ", "").replace("\\_", "_") + '")'

    typ = re.sub(r"upright\(((?:[A-Za-z]|\\_)(?: ?(?:[A-Za-z]|\\_))+)\)", join, typ)
    if re.search(r"upright\([^\"()]", typ):
        fail("an upright(...) math name was not converted to a quoted word")
    typ = re.sub(r"\bmono\(", "tt(", typ)

    # Typst sets a subscript on a relation symbol underneath it. The source
    # notation (\\preceq_k, \\parallel_k) is an ordinary subscript.
    typ, n_rel = re.subn(r"\b(prec\.curly\.eq(?:\.not)?|parallel)_", r"scripts(\1)_", typ)
    # \\mu\\text{M}: close the gap Typst leaves between the symbol and the unit text.
    n_mu = typ.count('mu upright("')
    typ = typ.replace('mu upright("', 'mu#h(0pt)upright("')
    # a.\\mathrm{ep}: no gap between the member dot and the upright name.
    n_dot = typ.count(' . upright("')
    typ = typ.replace(' . upright("', '.#h(0pt)upright("')
    if n_rel != 23 or n_mu != 2 or n_dot != 4:
        fail(f"math touch-ups matched {n_rel} relation subscripts, {n_mu} unit prefixes and {n_dot} member dots; "
             "expected 23, 2 and 4")

    # A display equation that is a long tuple of code names does not fit on one
    # line. Break it after the comma nearest the middle and align both lines
    # after the "=" sign. Layout only: no symbol is added or removed.
    sep = "\\,med "

    def wrap_tuple(m: re.Match) -> str:
        eq = m.group(1)
        names = re.findall(r'tt\("([^"]*)"\)', eq)
        if sum(len(n) for n in names) <= 75 or eq.count("=") != 1 or sep not in eq:
            return m.group(0)
        lhs, rhs = (side.strip() for side in eq.split("="))
        parts = rhs.split(sep)
        lines, cur, width = [], [], 0
        for part in parts:
            if cur and width + len(part) > 70:
                lines.append(cur)
                cur, width = [], 0
            cur.append(part)
            width += len(part)
        lines.append(cur)
        wrap_tuple.count += 1
        return ("$ " + lhs + " = & "
                + "\\, \\ & ".join(sep.join(line) for line in lines) + " $")

    wrap_tuple.count = 0
    typ = re.sub(r"(?m)^\$ (.*) \$$", wrap_tuple, typ)
    if wrap_tuple.count != 2:
        fail(f"expected to wrap exactly two overwide display equations, wrapped {wrap_tuple.count}")

    head = re.compile(r"#table\(\n    columns: (\d+),\n")
    found = head.findall(typ)
    if len(found) != len(colspecs):
        fail(f"{len(found)} tables in the Typst body, {len(colspecs)} in the Markdown")
    specs = iter(colspecs)

    def set_columns(m: re.Match) -> str:
        spec = next(specs)
        if spec.count(",") != int(m.group(1)):
            fail("column count mismatch between Markdown and Typst table")
        return f"#table(\n    columns: {spec},\n    table.hline(stroke: 0.9pt),\n"

    typ = head.sub(set_columns, typ)
    n_mid = typ.count("    table.hline(),\n")
    typ = typ.replace("    table.hline(),\n", "    table.hline(stroke: 0.5pt),\n")
    n_end = typ.count("\n  )]\n  , kind: table")
    typ = typ.replace("\n  )]\n  , kind: table", "\n    table.hline(stroke: 0.9pt),\n  )]\n  , kind: table")
    if n_mid != len(colspecs) or n_end != len(colspecs):
        fail("table rule insertion did not match every table")

    # Title: start the subtitle (the text after the first colon) on its own
    # line. Layout only; the document metadata keeps the one-line title.
    typ = re.sub(r"(?m)^(= [^\n:]*:) ", r"\1 #linebreak() ", typ, count=1)
    return typ


def typst_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> None:
    for tool in ("pandoc", "typst"):
        if shutil.which(tool) is None:
            fail(f"`{tool}` not found on PATH")
    for path in FIGURES.values():
        if not (HERE / path).is_file():
            fail(f"missing {path}; run make_figures.py first")
    md = SOURCE.read_text(encoding="utf-8")
    lines, colspecs, title = transform(md)
    proc = subprocess.run(
        ["pandoc", "-f", "markdown+autolink_bare_uris+lists_without_preceding_blankline", "-t", "typst-smart",
         "--wrap=none", "--columns=100000"],
        input="\n".join(lines).encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=HERE,
    )
    if proc.returncode != 0 or proc.stderr.strip():
        fail("pandoc: " + proc.stderr.decode("utf-8", "replace"))
    body = postprocess(proc.stdout.decode("utf-8"), colspecs)
    plain_title = re.sub(r"[`*]", "", title)
    # PDF author metadata comes from the manuscript's own Author line, if it has one.
    m = re.search(r"^\*\*Author:\*\* `?([^`\s]+)`?", md, flags=re.M)
    author = f"({typst_string(m.group(1))},)" if m else "()"
    preamble = (HERE / "template.typ").read_text(encoding="utf-8")
    doc = (
        f"// Generated by build.py from {SOURCE.name}. Do not edit by hand.\n"
        f"#set document(title: {typst_string(plain_title)}, author: {author}, date: none)\n\n"
        + preamble + "\n" + body
    )
    typ_path = HERE / f"{STEM}.typ"
    typ_path.write_text(doc, encoding="utf-8")
    pdf_path = HERE / f"{STEM}.pdf"
    proc = subprocess.run(
        ["typst", "compile", "--root", ".", "--ignore-system-fonts", typ_path.name, pdf_path.name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=HERE,
    )
    messages = (proc.stdout + proc.stderr).decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        fail("typst:\n" + messages)
    if messages:
        print("typst messages:\n" + messages)
    shutil.copyfile(pdf_path, HERE.parent / pdf_path.name)
    print(f"built {pdf_path.name} ({pdf_path.stat().st_size} bytes); typst messages: {'none' if not messages else 'see above'}")


if __name__ == "__main__":
    main()
