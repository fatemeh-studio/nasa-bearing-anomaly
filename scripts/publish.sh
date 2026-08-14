#!/usr/bin/env bash
#
# One command: check, test, render, publish. Never render without testing, never
# publish without rendering.
#
# The site is published from this machine rather than from CI, because Quarto renders
# .ipynb from the outputs stored in the notebook and does not execute it -- and CI
# starts from a clean clone with neither those outputs nor the 6.2 GB raw dataset.
# The cost of that choice is that nothing enforces the published site matching main.
# This script is what contains it.
#
# Usage:  ./scripts/publish.sh          check, test, render, then publish
#         ./scripts/publish.sh --dry    everything except the publish step
#
# `just publish` is the portfolio-standard name for this. `just` is not installed
# here and no sibling repository has a justfile, so this is the documented
# equivalent: it runs today, with no new dependency to install first.

set -euo pipefail

cd "$(dirname "$0")/.."

DRY=0
[[ "${1:-}" == "--dry" ]] && DRY=1

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# ── 1. Notebook outputs, before anything else ──────────────────────────────
# This is the failure that is quiet and total. Quarto renders stored outputs; if the
# working copy has been stripped, every notebook publishes as a code listing with no
# results -- the exact impression the site exists to prevent -- and nothing errors.
# Committing a notebook is what strips it: the nbstripout clean filter empties the
# blob, and a partially-staged file gets reverted to that state by pre-commit's
# stash cycle. So this is checked on every publish, not assumed.
step "Notebook outputs present on disk"
missing=0
for nb in notebooks/*.ipynb; do
    n=$(grep -c '"output_type"' "$nb" || true)
    printf '   %-46s %s outputs\n' "$(basename "$nb")" "$n"
    if [[ "$n" -eq 0 ]]; then
        missing=1
    fi
done
if [[ "$missing" -eq 1 ]]; then
    echo
    echo "ERROR: at least one notebook has no stored outputs on disk." >&2
    echo "Publishing now would put an empty notebook on the site." >&2
    echo "Fix: open it in Jupyter and run restart-and-run-all, then re-run this." >&2
    exit 1
fi

# ── 2. The numbers about to be published ───────────────────────────────────
# Printed rather than checked. The README and the site carry the same headline
# figures by design, and the rule keeping them together is that they change in the
# same commit -- so the useful thing here is to put the numbers in front of whoever
# is publishing, not to assert them.
step "Headline numbers in this build"
if [[ -f results/reports/business_summary.csv ]]; then
    column -s, -t < results/reports/business_summary.csv | cut -c1-120
else
    echo "ERROR: results/reports/business_summary.csv is missing." >&2
    echo "Run notebook 04, or: python -m nasa_bearing_anomaly.business --test all" >&2
    exit 1
fi

# ── 3. Gates ───────────────────────────────────────────────────────────────
step "ruff check"
ruff check .

step "ruff format --check"
ruff format --check .

step "pytest"
pytest -q

# ── 4. Render ──────────────────────────────────────────────────────────────
step "quarto render"
quarto render

# ── 5. Publish ─────────────────────────────────────────────────────────────
if [[ "$DRY" -eq 1 ]]; then
    step "Dry run -- stopping before publish"
    echo "   Site built in _site/. Open _site/index.html in a private window."
    exit 0
fi

step "quarto publish gh-pages"

# The first publish has to be interactive. Quarto needs to confirm creating the
# gh-pages branch and write _publish.yml recording the target; --no-prompt makes it
# refuse rather than ask, and the error it prints ("use first `quarto publish
# gh-pages` locally") describes the wrong cause. Once _publish.yml exists and the
# remote branch is there, every later publish is non-interactive.
if [[ -f _publish.yml ]] && git ls-remote --exit-code --heads origin gh-pages >/dev/null 2>&1; then
    quarto publish gh-pages --no-prompt
else
    echo "   First publish for this repository, so this step is interactive."
    echo "   Answering the prompt creates the gh-pages branch and writes"
    echo "   _publish.yml. Commit that file afterwards -- it is configuration,"
    echo "   not build output. Later runs need no prompt."
    echo
    quarto publish gh-pages
fi

cat <<'DONE'

Published. Two things that are not automated:

  1. Open the site in a PRIVATE window and confirm every figure is present,
     including inside the rendered notebooks. An empty notebook page means the
     outputs were stripped on disk after all.
  2. First publish only: Settings -> Pages -> Deploy from a branch -> gh-pages
     -> / (root), and set About -> Website to the site URL.
DONE
