#!/usr/bin/env python3
"""
compute_directory_sre.py
-------------------------
Compute exact Stabilizer Rényi Entropy (SRE) for all state trajectories
across dataset .pkl files in a target directory using HadaMAG.jl (CUDA-accelerated).

Usage:
    python3 compute_directory_sre.py <directory_path>

Example:
    python3 compute_directory_sre.py data_zip_7
    python3 compute_directory_sre.py correct_data
"""

import os
import sys
import pickle
import re
import numpy as np

# ---------------------------------------------------------
# 1. Initialize Julia HadaMAG.jl CUDA SRE Bridge
# ---------------------------------------------------------
print("Initializing Julia/HadaMAG.jl CUDA Bridge...")
try:
    from juliacall import Main as jl
    jl.seval("using HadaMAG, CUDA")
    jl.seval("""
    function jl_compute_sre_exact(state_vec, alpha, n, d)
        psi = StateVec(ComplexF64.(state_vec))
        s = SRE(psi, Float64(alpha))
        return (s, 0.0)
    end
    """)
    print("✅ Julia/HadaMAG.jl CUDA initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize Julia/HadaMAG.jl: {e}")
    sys.exit(1)


def compute_sre_exact(psi_np, alpha=2.0):
    """Compute exact SRE using HadaMAG.jl via JuliaCall."""
    try:
        arr_np = np.asarray(psi_np, dtype=np.complex128)
        norm_val = np.linalg.norm(arr_np)
        if norm_val > 1e-12:
            arr_np = arr_np / norm_val
        dim = len(arr_np)
        n_qubits = int(np.log2(dim))

        res = jl.jl_compute_sre_exact(arr_np, alpha, n_qubits, dim)
        val = float(res[0])
        return val if val != 0.0 else 1e-15, float(res[1])
    except Exception as err:
        print(f"  ⚠️ Error computing SRE: {err}")
        return 1e-15, 0.0


def parse_gap_sre(filename):
    """Parse gap k (0..15) from filename."""
    _match = re.search(r'(?:gap\s*|k\s*=\s*|_)?(\d+)\.pkl$', filename or "")
    if _match:
        val = float(_match.group(1))
        if val <= 15:
            return val
    return 0.0


def process_directory(target_dir):
    if not os.path.exists(target_dir):
        print(f"❌ Directory not found: {target_dir}")
        return

    pkl_files = sorted([
        f for f in os.listdir(target_dir) if f.endswith(".pkl")
    ])

    if not pkl_files:
        print(f"⚠️ No .pkl files found in {target_dir}")
        return

    print(f"\n📂 Processing {len(pkl_files)} dataset files in directory: {target_dir}\n" + "=" * 60)

    for idx, fname in enumerate(pkl_files, 1):
        fpath = os.path.join(target_dir, fname)
        derived_gap = parse_gap_sre(fname)
        print(f"[{idx}/{len(pkl_files)}] File: {fname} (Derived Initial Gap SRE = {derived_gap})")

        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"  ❌ Error reading {fname}: {e}")
            continue

        if "final_states" not in data:
            print(f"  ⚠️ 'final_states' key missing in {fname}, skipping.")
            continue

        final_states = data["final_states"]
        num_trajectories = len(final_states)
        sre_data = data.get("sre", [None] * num_trajectories)

        updated_count = 0
        for j in range(num_trajectories):
            state = final_states[j]
            traj = sre_data[j] if sre_data[j] is not None else [0.0, 0.0]

            if not isinstance(traj, list):
                traj = list(traj)

            init_sre = traj[0]
            final_sre = traj[-1]
            changed = False

            # 1. Update initial SRE from gap if near 0
            if abs(init_sre) <= 1e-5 and derived_gap != 0.0:
                init_sre = derived_gap
                traj[0] = derived_gap
                changed = True

            # 2. Compute final exact SRE if near 0
            if abs(final_sre) <= 1e-5:
                computed_final, _ = compute_sre_exact(state, alpha=2.0)
                final_sre = computed_final
                traj[-1] = computed_final
                changed = True

            if changed:
                updated_count += 1
                sre_data[j] = traj

        data["sre"] = sre_data

        # Save back updated dataset file
        try:
            with open(fpath, "wb") as f:
                pickle.dump(data, f)
            print(f"  ✅ Saved updated SRE values for {updated_count}/{num_trajectories} trajectories in {fname}")
        except Exception as e:
            print(f"  ❌ Error saving updated file {fname}: {e}")

    print("\n🎉 Directory processing complete!\n")


if __name__ == "__main__":
    dir_to_process = sys.argv[1] if len(sys.argv) > 1 else "data_zip_7"
    process_directory(dir_to_process)
