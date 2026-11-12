#!/usr/bin/env python3
"""Automated checks on the compiled publication PDF.

Run from this directory, after build.py:
    PYTHONDONTWRITEBYTECODE=1 python3 -B audit_pdf.py

Needs PyMuPDF (`pymupdf`) and the `pandoc` executable. Exits non-zero if any
check fails. The checks are:

1. Fonts: only the Typst-bundled text fonts and the figure font are embedded.
2. Geometry: no text extends outside the type area of any page.
3. Characters: no replacement, soft-hyphen or zero-width character in the text,
   and no prime mark standing in for an apostrophe.
4. Required content: author placeholders, reference count, accented names and
   scientific symbols are present.
5. Number parity: the multiset of digit runs in the PDF body text equals that
   of the Markdown source (figure artwork and page numbers excluded).
6. Word parity: every word of the Markdown prose occurs at least as often in
   the PDF body text.
7. Metadata: no local path in the PDF metadata or text.
"""
from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
STEM = "CEG-JOURNAL-MANUSCRIPT-V3.1"
MM = 72 / 25.4
LEFT, RIGHT, TOP, BOTTOM = 22 * MM, 22 * MM, 22 * MM, 25 * MM
TOL = 1.0
DESCENT = 4.0  # ink (descenders, subscripts) may hang at most this far below the type area
failures: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def main() -> None:
    doc = pymupdf.open(HERE / f"{STEM}.pdf")
    md = (HERE.parent / f"{STEM}.md").read_text(encoding="utf-8")

    fonts = sorted({f[3].split("+")[-1] for p in doc for f in p.get_fonts()})
    # STIX is the matplotlib-bundled fallback for one script letter in Figure 2.
    allowed = ("NewCM", "DejaVuSans", "STIX")
    check("fonts", all(f.startswith(allowed) for f in fonts), f"{len(fonts)} embedded: {', '.join(fonts)}")

    body_text: list[str] = []
    overflow: list[str] = []
    figure_pages: list[int] = []
    for pno, page in enumerate(doc, start=1):
        w, h = page.rect.width, page.rect.height
        lines = [l for b in page.get_text("rawdict")["blocks"] for l in b.get("lines", [])]
        # Figure artwork is the only place the proportional DejaVu Sans face is used.
        art = [pymupdf.Rect(s["bbox"]) for l in lines for s in l["spans"]
               if s["font"].split("+")[-1].startswith("DejaVuSans") and "Mono" not in s["font"]
               and "".join(c["c"] for c in s["chars"]).strip()]
        fig = None
        if art:
            fig = art[0]
            for r in art[1:]:
                fig |= r
            fig = fig + (-12, -12, 12, 12)
            figure_pages.append(pno)
        for l in lines:
            chars = [c for s in l["spans"] for c in s["chars"]]
            text = "".join(c["c"] for c in chars)
            inked = [c for c in chars if c["c"].strip()]
            if not inked:
                continue
            r = pymupdf.Rect(min(c["bbox"][0] for c in inked), min(c["bbox"][1] for c in inked),
                             max(c["bbox"][2] for c in inked), max(c["bbox"][3] for c in inked))
            # A figure is a full-width block, so any line lying wholly inside its vertical
            # band is artwork. (Testing containment in the box around the proportional
            # labels alone misses a monospaced node name that is wider than every
            # proportional label, as in Figure 1.)
            if fig is not None and r.y0 >= fig.y0 and r.y1 <= fig.y1:
                continue
            if r.y0 > h - BOTTOM and re.fullmatch(r"\d+", text.strip()):
                continue  # page number
            # Typst measures the type area to the text baseline, so on a full page the
            # descenders of the last line hang below it. Test the line's baseline (the
            # most common glyph origin; subscripts sit lower), and bound the ink box by a
            # descender allowance so that a real overflow is still reported.
            origins = [round(c["origin"][1], 1) for c in inked]
            baseline = max(set(origins), key=origins.count)
            if r.x0 < LEFT - TOL or r.x1 > w - RIGHT + TOL or r.y0 < TOP - 10 or baseline > h - BOTTOM + TOL or r.y1 > h - BOTTOM + DESCENT:
                overflow.append(f"p{pno} {text.strip()[:40]!r} x=({r.x0:.1f},{r.x1:.1f}) y=({r.y0:.1f},{r.y1:.1f})")
            body_text.append(text)
        for d in page.get_drawings():
            r = d["rect"]
            if r.width > 5 and (r.x0 < LEFT - 3 or r.x1 > w - RIGHT + 3):
                overflow.append(f"p{pno} drawing x=({r.x0:.1f},{r.x1:.1f})")
    check("geometry", not overflow, f"{len(overflow)} items outside the type area on {len(doc)} pages"
          + ("; first: " + "; ".join(overflow[:6]) if overflow else ""))
    print(f"       figure artwork found on pages {figure_pages}")

    full = "\n".join(p.get_text() for p in doc)
    bad = {hex(ord(c)): full.count(c) for c in "\ufffd\u00ad\u200b\u200c\u200d\u2060\ufeff\ufffc" if c in full}
    # A prime is legitimate in mathematics (c′) but not as a possessive apostrophe.
    primes = len(re.findall("[0-9A-Za-z]\u2032s\\b", full))
    check("characters", not bad and not primes, f"forbidden characters: {bad or 'none'}; prime used as apostrophe: {primes}")

    required = ["Königshoff", "Schünemann", "Rodríguez-Abreu", "Csőszi", "Fülöp", "±", "×", "–", "µ", "²", "VO₂max"]
    missing = [t for t in required if t not in full]
    n_ph = len(re.findall(r"\[AUTHOR\s+PLACEHOLDER", full))
    n_ph_md = md.count("[AUTHOR PLACEHOLDER")
    check("required content", not missing and n_ph == n_ph_md,
          f"missing: {missing or 'none'}; author placeholders {n_ph} (source {n_ph_md})")

    # ---- parity against the Markdown source --------------------------------
    src = re.sub(r"```mermaid.*?```", "", md, flags=re.S)
    # TeX row-spacing arguments such as \\[4pt] are layout, not content.
    src = re.sub(r"\\\\\[\d+pt\]", r"\\\\", src)
    plain = subprocess.run(["pandoc", "-f", "markdown+lists_without_preceding_blankline", "-t", "plain", "--wrap=none"], input=src.encode("utf-8"),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout.decode("utf-8")
    pdf_body = norm("\n".join(body_text))
    num_src = Counter(re.findall(r"\d+", norm(plain)))
    num_pdf = Counter(re.findall(r"\d+", pdf_body))
    diff = {k: (num_src[k], num_pdf[k]) for k in sorted(set(num_src) | set(num_pdf), key=lambda x: (len(x), x))
            if num_src[k] != num_pdf[k]}
    check("number parity", not diff, f"{sum(num_src.values())} digit runs in source, {sum(num_pdf.values())} in PDF body; "
          f"differences (source, pdf): {diff or 'none'}")

    nomath = re.sub(r"\$\$.*?\$\$", " ", src, flags=re.S)
    nomath = re.sub(r"\$[^$\n]*\$", " ", nomath)
    plain_nm = subprocess.run(["pandoc", "-f", "markdown+lists_without_preceding_blankline", "-t", "plain", "--wrap=none"], input=nomath.encode("utf-8"),
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout.decode("utf-8")

    def words(s: str) -> Counter:
        s = norm(s).replace("\u2010", " ").replace("-", " ")
        return Counter(wd.lower() for wd in re.findall(r"[A-Za-z]+", s))

    w_src = words(plain_nm)
    w_pdf = words("\n".join(body_text))
    short = {k: (v, w_pdf[k]) for k, v in w_src.items() if w_pdf[k] < v}
    check("word parity", not short, f"{sum(w_src.values())} prose words in source; words under-represented in the PDF "
          f"(source, pdf): {dict(list(short.items())[:12]) or 'none'}")

    meta = " ".join(str(v) for v in doc.metadata.values() if v)
    # Home-directory patterns are assembled here so that this file itself contains no path literal.
    patterns = ["/" + "Users" + "/", "/" + "home" + "/", "C:" + chr(92), "file:" + "//"]
    leaks = [t for t in patterns if t in meta or t in full]
    check("metadata", not leaks, f"title={doc.metadata.get('title')!r}; author={doc.metadata.get('author')!r}; "
          f"creator={doc.metadata.get('creator')!r}; path leaks: {leaks or 'none'}")

    print(f"pages: {len(doc)}")
    if failures:
        sys.exit("FAILED: " + ", ".join(failures))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
