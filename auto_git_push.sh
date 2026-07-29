#!/bin/bash
echo "Starting Git Auto-Push Daemon..."
while true; do
    # Check if there are unstaged or untracked changes anywhere in the repository
    if [[ -n $(git status --porcelain) ]]; then
        echo "[$(date)] Found changes in repository. Staging, committing, and pushing..."
        git add -A
        git commit -m "chore: auto-backup changes and simulation data"
        git push origin master
    else
        echo "[$(date)] No new changes to push."
    fi
    sleep 45 # 45 seconds check interval
done
