#!/usr/bin/env python3
import os
import sys
import glob
import time
import shutil
import pickle
import json
import subprocess
from datetime import datetime

DOWNLOADS_DIR = "/mnt/c/Users/Naga/Downloads"
TARGET_DEST_DIR = "zip7"
STATE_FILE = ".processed_downloads.json"
CHECK_INTERVAL_SECONDS = 30 * 60  # 30 minutes

def load_processed_files():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return set(json.load(f))
        except Exception as e:
            print(f"[{get_timestamp()}] Warning reading state file: {e}")
    return set()

def save_processed_files(processed_set):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(list(processed_set), f, indent=2)
    except Exception as e:
        print(f"[{get_timestamp()}] Error saving state file: {e}")

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def unpack_pkl_file(archive_path, destination_dir):
    """
    Unpacks a combined .pkl archive back into separate files inside destination_dir.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    try:
        with open(archive_path, "rb") as f:
            packed_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading archive: {e}")
        return False

    if not isinstance(packed_data, dict):
        print(f"Warning: {archive_path} content is not a dict archive.")
        return False

    # Extract each file content back to disk
    for filename, content in packed_data.items():
        output_path = os.path.join(destination_dir, filename)
        try:
            with open(output_path, "wb") as f:
                pickle.dump(content, f)
            print(f"Unpacked: {filename}")
        except Exception as e:
            print(f"Error unpacking {filename}: {e}")

    print(f"\nSuccessfully unpacked all files into directory: {destination_dir}")
    return True

def push_to_git(filename):
    print(f"[{get_timestamp()}] Staging and pushing changes to git...")
    try:
        subprocess.run(["git", "add", "."], check=True)
        commit_msg = f"feat(data): unpack new pkl file '{filename}' into {TARGET_DEST_DIR} [{get_timestamp()}]"
        res = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
        if "nothing to commit" in res.stdout or "nothing to commit" in res.stderr:
            print(f"[{get_timestamp()}] Nothing new to commit.")
        else:
            print(f"[{get_timestamp()}] Git commit successful: {commit_msg}")
            subprocess.run(["git", "push", "origin", "master"], check=True)
            print(f"[{get_timestamp()}] Git push completed successfully.")
    except Exception as e:
        print(f"[{get_timestamp()}] Error pushing to git: {e}")

def is_file_ready(file_path):
    if os.path.exists(file_path + ".crdownload") or os.path.exists(file_path + ".tmp"):
        return False
    try:
        size1 = os.path.getsize(file_path)
        time.sleep(1.5)
        size2 = os.path.getsize(file_path)
        return size1 == size2 and size1 > 0
    except Exception:
        return False

def check_and_process():
    processed = load_processed_files()
    pkl_files = sorted(glob.glob(os.path.join(DOWNLOADS_DIR, "*.pkl")))
    
    new_found = False
    current_repo_dir = os.getcwd()

    for src_path in pkl_files:
        filename = os.path.basename(src_path)
        # Use filename as unique identifier
        if filename in processed:
            continue

        if not is_file_ready(src_path):
            print(f"[{get_timestamp()}] Skipping {filename} (file download currently in progress)...")
            continue

        print(f"\n[{get_timestamp()}] Found new pkl file: {filename}")
        dest_path = os.path.join(current_repo_dir, filename)

        # 1. Copy file to current directory
        print(f"[{get_timestamp()}] Copying {src_path} -> {dest_path}...")
        try:
            shutil.copy2(src_path, dest_path)
            print(f"[{get_timestamp()}] Copy complete.")
        except Exception as e:
            print(f"[{get_timestamp()}] Failed to copy {filename}: {e}")
            continue

        # 2. Unpack pkl file into zip7
        print(f"[{get_timestamp()}] Unpacking {filename} into {TARGET_DEST_DIR}...")
        unpacked_ok = unpack_pkl_file(dest_path, TARGET_DEST_DIR)

        # 3. Push to git
        push_to_git(filename)

        # Mark as processed
        processed.add(filename)
        save_processed_files(processed)
        new_found = True

    if not new_found:
        print(f"[{get_timestamp()}] No new .pkl files found in {DOWNLOADS_DIR}.")

def main():
    run_once = "--once" in sys.argv
    print(f"[{get_timestamp()}] Starting Download Watcher Service...")
    print(f"  - Source folder: {DOWNLOADS_DIR}")
    print(f"  - Target directory: {TARGET_DEST_DIR}")
    print(f"  - Interval: {CHECK_INTERVAL_SECONDS // 60} minutes")
    print(f"  - Mode: {'Single Run (--once)' if run_once else 'Continuous Loop'}")

    if run_once:
        check_and_process()
    else:
        while True:
            check_and_process()
            print(f"\n[{get_timestamp()}] Sleeping for {CHECK_INTERVAL_SECONDS // 60} minutes...")
            time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
