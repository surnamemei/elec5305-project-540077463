#!/usr/bin/env bash
# Build the final report from report/FINAL_REPORT.pandoc.md.
#
# Outputs (repository root):
#   FINAL_REPORT.md   GitHub-readable Markdown with numbered IEEE citations
#   FINAL_REPORT.pdf  PDF (requires xelatex)
#
# Requirements: pandoc >= 3 (set PANDOC=/path/to/pandoc if not on PATH),
#               xelatex with the Linux Libertine O and DejaVu Math TeX Gyre fonts.
#
# Run from the repository root:
#   bash report/build_report.sh

set -euo pipefail

PANDOC="${PANDOC:-pandoc}"
SOURCE="report/FINAL_REPORT.pandoc.md"
COMMON=(
  --citeproc
  --bibliography=FINAL_REPORT_REFERENCES.bib
  --csl=report/ieee.csl
  --resource-path=.
  --metadata=link-citations:true
)

# GitHub Markdown: title block written explicitly (no YAML front matter)
{
  echo "# Why Is Wav2Vec2 Robust to Lossy Audio Compression? A Signal- and Representation-Level Study of MP3, Opus and EnCodec"
  echo
  echo "**Jinghang Mei (SID 540077463)** — ELEC5305, The University of Sydney  "
  echo "Code and results: https://github.com/surnamemei/elec5305-project-540077463"
  echo
  echo "*Generated from \`report/FINAL_REPORT.pandoc.md\` by \`report/build_report.sh\`; a typeset version is in \`FINAL_REPORT.pdf\`.*"
  echo
  "$PANDOC" "$SOURCE" "${COMMON[@]}" \
    --to=gfm+tex_math_dollars \
    --wrap=none
} > FINAL_REPORT.md

"$PANDOC" "$SOURCE" "${COMMON[@]}" \
  --pdf-engine=xelatex \
  --include-in-header=report/header.tex \
  --variable=geometry:margin=2cm \
  --variable=fontsize:10pt \
  --variable=mainfont:"Linux Libertine O" \
  --variable=mathfont:"DejaVu Math TeX Gyre" \
  --variable=colorlinks:true \
  --output=FINAL_REPORT.pdf

echo "Built FINAL_REPORT.md and FINAL_REPORT.pdf"
