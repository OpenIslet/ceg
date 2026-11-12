// Layout template for the CEG publication PDF.
// build.py prepends this file to the pandoc-converted manuscript body.
// Fonts are the ones bundled with the Typst compiler, so the build does not
// depend on fonts installed on the machine.

#let body-font = "New Computer Modern"
#let math-font = "New Computer Modern Math"
#let mono-font = "DejaVu Sans Mono"

#set page(
  paper: "a4",
  margin: (x: 22mm, top: 22mm, bottom: 25mm),
  numbering: "1",
  number-align: center,
)
// Hyphenation is off so that no soft hyphen is written into the PDF text, and
// punctuation does not hang into the margin, so the type area is strict.
#set text(font: body-font, size: 10pt, lang: "en", region: "GB", hyphenate: false, overhang: false)
// pandoc writes typographic quotes itself; Typst must not re-guess them
// (it would turn the apostrophe after a digit into a prime mark).
#set smartquote(enabled: false)
#set par(justify: true, leading: 0.62em, spacing: 0.9em)
#show math.equation: set text(font: math-font)
#show math.equation.where(block: true): set block(above: 1em, below: 1em)

// Code spans: monospace, and breakable after "_" and "/" by means of a
// zero-width inline box. No invisible character is written into the PDF text.
#show raw: set text(font: mono-font, size: 0.85em)
// A command-line flag such as -B or --check is kept on one line.
#show raw.where(block: false): it => {
  show regex("\\B-{1,2}\\w+"): box
  show regex("[_/]"): m => m + box(width: 0pt)
  it
}
// \texttt{...} inside mathematics.
#let tt(s) = text(font: mono-font, size: 0.85em, s)

// Section rules in the Markdown source are not drawn.
#let divider() = v(0pt)

#show heading: set text(font: body-font)
#show heading.where(level: 2): it => block(above: 1.7em, below: 0.85em, text(size: 12.5pt, weight: "bold", it.body))
#show heading.where(level: 3): it => block(above: 1.4em, below: 0.75em, text(size: 10.5pt, weight: "bold", it.body))

#set list(indent: 0.5em, body-indent: 0.5em, marker: ([•], [–]))
#set enum(indent: 0.5em, body-indent: 0.5em)

// Tables: booktabs rules only, smaller type, left-ragged cells.
#set table(stroke: none, inset: (x: 4.5pt, y: 3pt))
#show table: set text(size: 8pt)
#show table: set par(justify: false, leading: 0.5em)
#show table.cell.where(y: 0): set text(weight: "bold")
// Inline formulas in table cells are not split across lines.
#show table: it => {
  show math.equation.where(block: false): box
  it
}
// Every table fits on one page, so tables are kept whole (figures are
// unbreakable by default) and no header row is repeated.
#show figure.where(kind: table): set block(above: 0.55em, below: 1.4em)

#let tablecaption(body) = block(sticky: true, above: 1.4em, below: 0.55em, width: 100%, {
  set par(justify: false)
  set text(size: 9pt)
  body
})

#let figimage(path) = align(center, image(path, width: 100%))
// Figures float to the top or bottom of a page so the text is not interrupted
// by a blank area when a figure does not fit where it is first mentioned.
#let figblock(body) = place(auto, float: true, clearance: 1.6em, block(width: 100%, {
  set align(left)
  set text(size: 9pt)
  set par(justify: true, leading: 0.55em)
  body
}))

#let frontmatter(body) = {
  set align(center)
  set par(justify: false, leading: 0.6em, spacing: 0.9em)
  show heading.where(level: 1): it => block(above: 0em, below: 1.3em, text(size: 17pt, weight: "bold", hyphenate: false, it.body))
  set text(size: 9.5pt)
  body
  v(0.6em)
}

#let abstractblock(body) = block(width: 100%, inset: (x: 1.4em), {
  set text(size: 9.5pt)
  show heading.where(level: 2): it => block(above: 0.4em, below: 0.7em, align(center, text(size: 11pt, weight: "bold", it.body)))
  body
})

#let references(body) = {
  set text(size: 8.6pt)
  set par(justify: false, leading: 0.52em)
  set enum(numbering: "1.", body-indent: 0.6em, spacing: 0.62em)
  body
}

