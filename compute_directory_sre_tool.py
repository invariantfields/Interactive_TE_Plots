#!/usr/bin/env python3
"""
compute_directory_sre_tool.py
------------------------------
Standalone tool to compute EXACT SRE values (via HadaMAG.jl + CUDA) for
all final_states in every .pkl file within a given directory.

For each seed j in each file:
  - init_sre  = stabilizer gap k (exact, parsed from filename)
  - final_sre = HadaMAG.SRE( final_states[j], alpha=2, backend=CUDA )

The 'sre' key is rewritten as a 2-element list [init_sre, final_sre]
reflecting the true computed endpoints.

Usage:
  python3 compute_directory_sre_tool.py --dir data_zip_7
  python3 compute_directory_sre_tool.py --dir correct_data --batch-size 50
"""

import os
import sys
import re
import argparse
import pickle
import time
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_gap_from_filename(filename: str) -> float:
    """Extract stabilizer gap k from filename. Returns float (e.g. 3.0 for _stps_3.pkl)."""
    match = re.search(r"(?:gap|stps|steps)[_\s]*(\d+)\.pkl$", filename, re.IGNORECASE)
    if not match:
        match = re.search(r"_(\d+)\.pkl$", filename)
    if match:
        val = float(match.group(1))
        if val <= 20:
            return val
    return 0.0


def init_julia():
    """Load Julia + HadaMAG and define the batch SRE function. Returns jl handle."""
    print("Initialising Julia / HadaMAG.jl CUDA bridge …")
    from juliacall import Main as jl

    jl.seval("using Logging; disable_logging(Logging.Error)")
    jl.seval("using HadaMAG, CUDA")
    jl.seval("using LinearAlgebra: norm")
    jl.seval("""
    function jl_compute_sre_batch(psi_batch_np, alpha, n_qubits, dim, num_states)
        results = zeros(Float64, num_states)
        for i in 1:num_states
            psi_row = psi_batch_np[i, :]
            psi_jl  = Vector{ComplexF64}(psi_row)
            nrm     = norm(psi_jl)
            if nrm > 1e-12
                psi_jl ./= nrm
            end
            psi_sv     = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, Int(n_qubits), Int(dim))
            sre_result = SRE(psi_sv, Int(round(alpha)); progress=false)
            results[i] = Float64(sre_result[1])
        end
        return results
    end
    """)
    print("  ✅ Julia HadaMAG batch SRE ready.")
    return jl


def compute_sre_batch(jl, states: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Compute exact SRE (alpha=2) for a batch of state vectors.

    Parameters
    ----------
    states : np.ndarray, shape (N, dim), complex128
    n_qubits : int

    Returns
    -------
    sre_values : np.ndarray, shape (N,), float64
    """
    n_states, dim = states.shape
    sre_jl = jl.jl_compute_sre_batch(states, 2, n_qubits, dim, n_states)
    return np.array(sre_jl, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_directory(directory_path: str, batch_size: int = 100):
    if not os.path.exists(directory_path):
        print(f"Error: directory '{directory_path}' does not exist.")
        sys.exit(1)

    files = sorted(
        [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith(".pkl")
        ]
    )

    if not files:
        print(f"No .pkl files found in '{directory_path}'.")
        return

    print("=" * 60)
    print(f"Directory : {directory_path}")
    print(f"Files     : {len(files)}")
    print(f"Batch size: {batch_size} states per Julia call")
    print("=" * 60)

    # Initialise Julia once, shared across all files
    jl = init_julia()

    for file_idx, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        gap_k = parse_gap_from_filename(filename)

        print(f"\n[{file_idx}/{len(files)}] {filename}  (gap k={gap_k:.0f})")

        with open(file_path, "rb") as fh:
            data = pickle.load(fh)

        final_states = data.get("final_states", [])
        if not final_states:
            print("  ⚠️  No 'final_states' key – skipping.")
            continue

        num_seeds = len(final_states)
        dim = len(final_states[0])
        n_qubits = int(round(np.log2(dim)))

        print(f"  Seeds: {num_seeds}, dim: {dim}, qubits: {n_qubits}")

        # ---- compute final SRE in batches ----------------------------------
        final_sre_all = np.zeros(num_seeds, dtype=np.float64)
        t0 = time.time()

        for start in range(0, num_seeds, batch_size):
            end = min(start + batch_size, num_seeds)
            batch = np.array(final_states[start:end], dtype=np.complex128)

            # Normalise
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            batch /= norms

            sre_batch = compute_sre_batch(jl, batch, n_qubits)
            final_sre_all[start:end] = sre_batch

            elapsed = time.time() - t0
            pct = (end / num_seeds) * 100
            rate = end / elapsed if elapsed > 0 else 1
            eta = (num_seeds - end) / rate
            print(
                f"  [{end:>5}/{num_seeds}] {pct:5.1f}%  "
                f"mean SRE={float(np.mean(sre_batch)):.4f}  "
                f"elapsed={elapsed:.1f}s  ETA={eta:.0f}s"
            )

        # ---- write updated sre trajectories --------------------------------
        # Two honest real endpoints: [init_sre, final_sre]
        updated_sre = [[gap_k, float(final_sre_all[j])] for j in range(num_seeds)]
        data["sre"] = updated_sre

        with open(file_path, "wb") as fh:
            pickle.dump(data, fh)

        total_elapsed = time.time() - t0
        print(
            f"  ✅ Saved – mean final SRE = {float(np.mean(final_sre_all)):.4f}  "
            f"({total_elapsed:.1f}s total)"
        )

    print(f"\n🎉  Done – processed all {len(files)} files in '{directory_path}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute exact HadaMAG SRE for all .pkl files in a directory."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing .pkl files (e.g. data_zip_7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of states per Julia SRE batch call (default: 100)",
    )
    args = parser.parse_args()
    process_directory(args.dir, batch_size=args.batch_size)
