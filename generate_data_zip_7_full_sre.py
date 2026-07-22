#!/usr/bin/env python3
"""
generate_data_zip_7_full_sre.py
---------------------------------
Generate 7-qubit, 1000-step, 1000-seed optimization datasets for all gaps k in {0..7}
with EXACT 101-step CUDA SRE trajectories computed at every optimization chunk via HadaMAG.jl.

Output Directory: data_zip_7/
Output Filenames: 7_qbt_1000_sds_ptmzng_jfr_1000_stps_{k}.pkl
"""

import os
import sys
import time
import pickle
import re
from itertools import combinations
import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import LBFGS

# GPU / JAX Configurations
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
jax.config.update("jax_enable_x64", True)


def generate_random_generators_symplectic(n_qubits: int, depth_multiplier: int = 10) -> list[str]:
    x_mat = np.zeros((n_qubits, n_qubits), dtype=int)
    z_mat = np.eye(n_qubits, dtype=int)
    r = np.zeros(n_qubits, dtype=int)

    def apply_H(target):
        r[:] ^= x_mat[:, target] & z_mat[:, target]
        x_mat[:, target], z_mat[:, target] = z_mat[:, target].copy(), x_mat[:, target].copy()

    def apply_S(target):
        r[:] ^= x_mat[:, target] & z_mat[:, target]
        z_mat[:, target] ^= x_mat[:, target]

    def apply_CNOT(control, target):
        r[:] ^= (x_mat[:, control] & z_mat[:, target]) & (x_mat[:, target] ^ z_mat[:, control] ^ 1)
        x_mat[:, target] ^= x_mat[:, control]
        z_mat[:, control] ^= z_mat[:, target]

    num_gates = depth_multiplier * n_qubits**2
    for _ in range(num_gates):
        gate = np.random.choice(["H", "S", "CNOT"])
        if gate == "H":
            apply_H(np.random.randint(n_qubits))
        elif gate == "S":
            apply_S(np.random.randint(n_qubits))
        elif n_qubits - 2 >= 0:
            c, t = np.random.choice(n_qubits, 2, replace=False)
            apply_CNOT(c, t)

    generators = []
    for i in range(n_qubits):
        sign = "-" if r[i] else "+"
        pauli_str = sign
        for j in range(n_qubits):
            x, z = x_mat[i, j], z_mat[i, j]
            if x == 0 and z == 0:
                pauli_str += "I"
            elif x == 1 and z == 0:
                pauli_str += "X"
            elif x == 1 and z == 1:
                pauli_str += "Y"
            elif x == 0 and z == 1:
                pauli_str += "Z"
        generators.append(pauli_str)
    return generators


def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    phase = -1.0 if pauli_str[0] == "-" else 1.0
    paulis = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    mat = paulis[pauli_str[1]]
    for p in pauli_str[2:]:
        mat = np.kron(mat, paulis[p])
    return phase * mat


def rand_Almost_Stab_state(n: int, gap: int) -> np.ndarray:
    dim = 2**n
    k = n - gap
    if k == 0:
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        return psi / np.linalg.norm(psi)

    gens = generate_random_generators_symplectic(n)
    chosen_gens = gens[:k]
    proj = np.eye(dim, dtype=complex)
    for g in chosen_gens:
        g_mat = pauli_string_to_matrix(g)
        proj = proj @ (np.eye(dim, dtype=complex) + g_mat) / 2.0

    raw_psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi = proj @ raw_psi
    norm = np.linalg.norm(psi)
    if norm < 1e-8:
        return rand_Almost_Stab_state(n, gap)
    return psi / norm


def run_jax_gpu_optimization(n: int, num_starts: int, num_loops: int, gap: int, step_mes: int = 10, filename: str = ""):
    n_dim = 2**n
    k_sub = n - n // 2
    combos = list(combinations(range(n), k_sub))

    def get_perms(n, k, combo):
        return [x for x in range(n) if x not in combo] + list(combo)

    perms_list = [jnp.array(get_perms(n, k_sub, c)) for c in combos]

    @jax.jit
    def get_purity_and_violation(psi_vec):
        psi = psi_vec[:n_dim] + 1j * psi_vec[n_dim:]
        psi /= jnp.linalg.norm(psi)
        psi_tensor = psi.reshape((2,) * n)

        rhos = [
            psi_tensor.transpose(perm).reshape(-1, 2**k_sub).conj().T 
            @ psi_tensor.transpose(perm).reshape(-1, 2**k_sub) 
            for perm in perms_list
        ]
        batch_rho = jnp.stack(rhos, axis=0)
        ex = jnp.linalg.eigvalsh(batch_rho)

        purities = jnp.sum(ex**2, axis=-1)
        avg_purity = jnp.mean(purities)
        max_purity = jnp.max(purities)

        rhs = ex[:, 1] + 2 * jnp.sqrt(jnp.maximum(ex[:, 0] * ex[:, 2], 1e-15))
        viols = jnp.maximum(0.0, ex[:, -1] - rhs)
        total_violation = jnp.sum(viols**2)

        return avg_purity, max_purity, total_violation

    @jax.jit
    def objective(params):
        _, _, total_viol = get_purity_and_violation(params)
        return total_viol

    solver = LBFGS(fun=objective, maxiter=step_mes, tol=1e-11)
    
    print("Initializing Julia/HadaMAG.jl CUDA Bridge...")
    from juliacall import Main as jl
    jl.seval("using Logging; disable_logging(Logging.Error)")
    jl.seval("using HadaMAG, CUDA")
    jl.seval("using LinearAlgebra: norm")
    jl.seval("""
    function jl_compute_sre_batch(psi_batch_np, alpha, n_qubits, dim, num_starts)
        results = zeros(Float64, num_starts)
        for i in 1:num_starts
            psi_row = psi_batch_np[i, :]
            psi_jl = Vector{ComplexF64}(psi_row)
            nrm = norm(psi_jl)
            if nrm > 1e-12
                psi_jl ./= nrm
            end
            psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, Int(n_qubits), Int(dim))
            sre_res = SRE(psi_sv, Int(round(alpha)); progress=false)
            results[i] = Float64(sre_res[1])
        end
        return results
    end
    """)
    print("Julia HadaMAG batch SRE initialized.")

    print(f"Generating {num_starts} initial states for Gap {gap}...")
    init_params_list = []
    for st in range(num_starts):
        psi_init = rand_Almost_Stab_state(n, gap)
        psi_np = np.array(psi_init)
        p = np.concatenate([np.real(psi_np), np.imag(psi_np)])
        init_params_list.append(p)
    params_batch = jnp.array(init_params_list, dtype=jnp.float64)

    vmapped_metrics = jax.vmap(get_purity_and_violation)
    vmapped_run = jax.vmap(solver.run)

    traj_avg_p = [[] for _ in range(num_starts)]
    traj_max_p = [[] for _ in range(num_starts)]
    traj_viol = [[] for _ in range(num_starts)]
    traj_sre = [[] for _ in range(num_starts)]

    start_time = time.time()

    avg_p, max_p, viol = vmapped_metrics(params_batch)
    avg_p_cpu, max_p_cpu, viol_cpu = np.array(avg_p), np.array(max_p), np.array(viol)
    
    params_np = np.array(params_batch)
    states_complex_batch = params_np[:, :n_dim] + 1j * params_np[:, n_dim:]
    sre_vals = jl.jl_compute_sre_batch(states_complex_batch, 2, n, n_dim, num_starts)
    sre_vals_cpu = np.array(sre_vals)

    for i in range(num_starts):
        traj_avg_p[i].append(float(avg_p_cpu[i]))
        traj_max_p[i].append(float(max_p_cpu[i]))
        traj_viol[i].append(float(viol_cpu[i]))
        traj_sre[i].append(float(sre_vals_cpu[i]))
    print(f"  Start Step | Mean Violation: {float(np.mean(viol_cpu)):.2e} | Mean SRE: {float(np.mean(sre_vals_cpu)):.4f}")

    num_chunks = max(1, num_loops // step_mes)

    for chunk_idx in range(1, num_chunks + 1):
        res = vmapped_run(params_batch)
        params_batch = res.params

        avg_p, max_p, viol = vmapped_metrics(params_batch)
        avg_p_cpu, max_p_cpu, viol_cpu = np.array(avg_p), np.array(max_p), np.array(viol)

        params_np = np.array(params_batch)
        states_complex_batch = params_np[:, :n_dim] + 1j * params_np[:, n_dim:]
        sre_vals = jl.jl_compute_sre_batch(states_complex_batch, 2, n, n_dim, num_starts)
        sre_vals_cpu = np.array(sre_vals)

        for i in range(num_starts):
            traj_avg_p[i].append(float(avg_p_cpu[i]))
            traj_max_p[i].append(float(max_p_cpu[i]))
            traj_viol[i].append(float(viol_cpu[i]))
            traj_sre[i].append(float(sre_vals_cpu[i]))

        if chunk_idx % 10 == 0 or chunk_idx == num_chunks:
            elapsed = time.time() - start_time
            rate = elapsed / chunk_idx
            remaining = rate * (num_chunks - chunk_idx)
            print(
                f"  Step {chunk_idx * step_mes:5d}/{num_loops} | "
                f"Mean Violation: {float(np.mean(viol_cpu)):.2e} | "
                f"Mean SRE: {float(np.mean(sre_vals_cpu)):.4f} | "
                f"Elapsed: {elapsed:.1f}s | Est. Remaining: {remaining:.1f}s"
            )

    final_params_np = np.array(params_batch)
    final_states_batch = final_params_np[:, :n_dim] + 1j * final_params_np[:, n_dim:]
    final_states_normalized = final_states_batch / np.linalg.norm(final_states_batch, axis=1, keepdims=True)

    final_states_list = [final_states_normalized[i] for i in range(num_starts)]

    trajectories = {
        "average_purity": traj_avg_p,
        "max_purity": traj_max_p,
        "sre": traj_sre,
        "total_violation": traj_viol,
        "final_states": final_states_list,
    }

    with open(filename, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"✅ Saved full 101-step SRE trajectory dataset to {filename}")
    return trajectories


def generate_all_data_zip_7():
    os.makedirs("data_zip_7", exist_ok=True)
    n_qubits = 7
    num_seeds = 1000
    num_steps = 1000
    chunk_size = 10

    print(f"=======================================================")
    print(f"Generating Full 101-Step SRE Datasets for data_zip_7/")
    print(f"7 Qubits | 1000 Seeds | 1000 Steps | Chunk size = {chunk_size}")
    print(f"=======================================================")

    for gap in range(0, n_qubits + 1):
        filename = f"data_zip_7/7_qbt_1000_sds_ptmzng_jfr_1000_stps_{gap}.pkl"
        print(f"\n🚀 [Gap {gap}/7] Computing full SRE trajectories -> {filename}")
        run_jax_gpu_optimization(
            n=n_qubits,
            num_starts=num_seeds,
            num_loops=num_steps,
            gap=gap,
            step_mes=chunk_size,
            filename=filename,
        )

    print("\n🎉 ALL data_zip_7 full SRE datasets generated successfully!")


if __name__ == "__main__":
    generate_all_data_zip_7()
