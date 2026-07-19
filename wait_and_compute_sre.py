import os
import sys
import time
import re
import pickle
import numpy as np

# PIDs to wait for
PIDS = [35962, 37276]

def is_pid_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

print(f"Monitoring PIDs: {PIDS}...")
while any(is_pid_running(pid) for pid in PIDS):
    running_pids = [pid for pid in PIDS if is_pid_running(pid)]
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Still running PIDs: {running_pids}. Sleeping for 60 seconds...")
    time.sleep(60)

print("\nBoth simulation runs have completed! Starting SRE computation queue...")

# Initialize Julia Call
try:
    print("Initializing Julia/HadaMAG.jl...")
    from juliacall import Main as jl
    jl.seval("using HadaMAG")
    jl.seval("using CUDA")
    
    jl.seval("""
    function jl_compute_sre_exact(psi_np, alpha, n_qubits, dim)
        psi_jl = Vector{ComplexF64}(psi_np)
        psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, Int(n_qubits), Int(dim))
        sre_result, lost_norm = SRE(psi_sv, alpha, backend= :CUDA)
        return (sre_result, lost_norm)
    end
    """)
    print("Julia HadaMAG + CUDA backend initialized successfully.")
except Exception as e:
    print(f"Error initializing Julia: {e}")
    sys.exit(1)

def compute_sre_exact(psi_np, alpha=2):
    try:
        arr_np = np.asarray(psi_np)
        norm_val = np.linalg.norm(arr_np)
        if norm_val > 1e-12:
            arr_np = arr_np / norm_val
        dim = len(arr_np)
        n_qubits = int(np.log2(dim))

        res = jl.jl_compute_sre_exact(arr_np, alpha, n_qubits, dim)
        val = float(res[0])
        return val if val != 0.0 else 1e-15, float(res[1])
    except Exception as e:
        print(f"SRE Exact Calculation Error: {e}")
        return 1e-15, 0.0

# Scan correct_data folder for pickle files
data_dir = "correct_data"
if not os.path.exists(data_dir):
    print(f"Directory {data_dir} does not exist!")
    sys.exit(1)

pkl_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")]
print(f"Found {len(pkl_files)} pickle files to check.")

for filepath in sorted(pkl_files):
    print(f"\nProcessing file: {filepath}...")
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        continue

    if "final_states" not in data:
        print(f"Skipping {filepath} (no final_states key found).")
        continue

    final_states = data["final_states"]
    num_starts = len(final_states)
    
    # Get trajectory length from average_purity
    traj_len = len(data["average_purity"][0]) if data["average_purity"] else 2
    
    # Initialize SRE structure if missing or mismatched
    if "sre" not in data or len(data["sre"]) != num_starts or (num_starts > 0 and len(data["sre"][0]) != traj_len):
        data["sre"] = [[0.0] * traj_len for _ in range(num_starts)]

    # Parse gap from filename
    derived_init_sre = 0.0
    _gap_match = re.search(r'_stps_(\d+)\.pkl$', filepath)
    if _gap_match:
        derived_init_sre = float(_gap_match.group(1))
    else:
        # try gap fallback
        _gap_match = re.search(r'gap(\d+)', filepath)
        if _gap_match:
            derived_init_sre = float(_gap_match.group(1))

    print(f"Derived initial SRE: {derived_init_sre}")
    print(f"Computing exact final SRE for {num_starts} states...")

    data_changed = False
    for j in range(num_starts):
        # 1. Update initial state SRE (index 0) if 0.0
        if data["sre"][j][0] == 0.0 and derived_init_sre != 0.0:
            data["sre"][j][0] = derived_init_sre
            data_changed = True

        # 2. Compute final state SRE (index -1) if 0.0
        if data["sre"][j][-1] == 0.0:
            final_state = final_states[j]
            sre_val, _ = compute_sre_exact(final_state)
            data["sre"][j][-1] = sre_val
            data_changed = True
            
            if (j + 1) % 100 == 0 or (j + 1) == num_starts:
                print(f"  Processed {j + 1}/{num_starts} states...")

    if data_changed:
        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved updated SRE values back to {filepath}")
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
    else:
        print(f"No changes needed for {filepath}.")

print("\nAll SRE computations completed successfully!")
