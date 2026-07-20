#!/usr/bin/env bash
# Auto-push script for the Interactive_TE_Plots repository.
# It adds all changes, creates a timestamped commit (if there are changes), and pushes to the remote.
# Intended to be run periodically (e.g., via cron) during long-running data generation.

set -euo pipefail

# Change to the repository root (assumes this script is placed in the repo root)
cd "$(dirname "$0")"

# Ensure we are on the desired branch (adjust if needed)
git checkout main || true

# Add all changes (including new data files)
git add -A

# Check if there is anything to commit
if git diff-index --quiet HEAD --; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] No changes to commit."
else
  commit_msg="Auto push $(date '+%Y-%m-%d %H:%M:%S')"
  git commit -m "$commit_msg"
  git push
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pushed changes: $commit_msg"
fi
