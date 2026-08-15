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

# ── 1. Clean working tree ──────────────────────────────────────────────────
# quarto publish runs `git stash` when the tree is dirty and then switches
# branches. Both rewrite notebooks from git, where the clean filter has already
# stripped the outputs -- which is what emptied all four mid-publish on
# 2026-08-14. On a clean tree quarto never stashes and never touches them.
#
# Checked with `git diff --quiet` rather than `git status`, which reports a
# notebook as modified whenever its stat cache is stale even though the filtered
# content is identical. That would block every run.
step "Working tree clean"
if git diff --quiet && git diff --cached --quiet; then
    echo "   nothing uncommitted"
elif [[ "$DRY" -eq 1 ]]; then
    echo "   WARNING: uncommitted changes. --dry continues, but a real publish"
    echo "   would stop here."
else
    echo
    echo "ERROR: uncommitted changes. Commit before publishing." >&2
    echo "quarto stashes a dirty tree and switches branches, which strips the" >&2
    echo "notebook outputs this site renders from." >&2
    echo >&2
    git status --short >&2
    exit 1
fi

# ── 2. Notebook outputs ────────────────────────────────────────────────────
# This is the failure that is quiet and total. Quarto renders stored outputs; if the
# working copy has been stripped, every notebook publishes as a code listing with no
# results -- the exact impression the site exists to prevent -- and nothing errors.
# Committing is what strips them, and not only the notebook being committed:
# measured 2026-08-14, if ANY file has unstaged changes then pre-commit stashes,
# resets the worktree, and EVERY notebook comes back stripped -- including ones
# untouched by the commit. Section 1 is what usually prevents that; this stays as
# the backstop, because rebases and branch switches strip them too.
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

# ── 3. The numbers about to be published ───────────────────────────────────
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

# ── 4. Gates ───────────────────────────────────────────────────────────────
step "ruff check"
ruff check .

step "ruff format --check"
ruff format --check .

step "pytest"
pytest -q

# ── 5. Render ──────────────────────────────────────────────────────────────
step "quarto render"
quarto render

# ── 6. Publish ─────────────────────────────────────────────────────────────
if [[ "$DRY" -eq 1 ]]; then
    step "Dry run -- stopping before publish"
    echo "   Site built in _site/. Open _site/index.html in a private window."
    exit 0
fi

step "quarto publish gh-pages"

# The first publish has to be interactive. Quarto refuses to create the gh-pages
# branch without confirmation, and --no-prompt turns that into an error whose text
# names the wrong cause ("use first `quarto publish gh-pages` locally" -- which is
# exactly what is running). The condition is the remote branch alone: quarto is
# documented to write _publish.yml on first publish and did not do so here, so
# testing for that file would keep every future run interactive.
#
# Note pre-commit must be installed with --allow-missing-config. The gh-pages
# branch holds only built HTML and carries no .pre-commit-config.yaml, so a plain
# install aborts the commit there and leaves the branch with no commits at all.
if git ls-remote --exit-code --heads origin gh-pages >/dev/null 2>&1; then
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
