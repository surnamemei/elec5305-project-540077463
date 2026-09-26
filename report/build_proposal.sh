#!/usr/bin/env bash
# Build "ELEC5305 Project Proposal v3.pdf" from report/proposal_v3.md.
# Requirements: pandoc >= 3 (set PANDOC=/path/to/pandoc if not on PATH) and xelatex.
# Run from the repository root:  bash report/build_proposal.sh
set -euo pipefail
PANDOC="${PANDOC:-pandoc}"
"$PANDOC" report/proposal_v3.md \
  --pdf-engine=xelatex \
  --include-in-header=report/proposal_header.tex \
  --variable=geometry:margin=2cm \
  --variable=fontsize:10pt \
  --variable=mainfont:"Linux Libertine O" \
  --variable=colorlinks:true \
  --output="ELEC5305 Project Proposal v3.pdf"
echo "Built ELEC5305 Project Proposal v3.pdf"
