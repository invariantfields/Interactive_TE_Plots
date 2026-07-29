# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "cupy-cuda13x",
#     "jax[cuda13]",
#     "jaxopt[cuda13]",
#     "matplotlib",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    import time
    import pickle
    import numpy as np
    from scipy.linalg import hadamard
    from itertools import combinations
    import jax
    import jax.numpy as jnp
    from jax import random, vmap
    import jaxopt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import marimo as mo

    # Configure GPU memory sharing
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70"
    jax.config.update("jax_enable_x64", True)

    return (
        combinations,
        go,
        hadamard,
        jax,
        jaxopt,
        jnp,
        make_subplots,
        mo,
        np,
        os,
        pickle,
        random,
        sys,
        time,
        vmap,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # 🚀 High-Performance GPU Trajectory Generation (Restored JAX Hildebrand Engine)
        ### Fast Walsh-Hadamard Transform (FWHT) SRE Engine | Optimized for RTX 6000 PRO (96GB VRAM)

        [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/invariantfields/Interactive_TE_Plots/blob/master/fast_7q_gpu_data_generation_notebook.py)

        This notebook implements **Restored JAX Hildebrand Spectral Violation Minimization** combined with **100% GPU Fast Walsh-Hadamard Transform (FWHT) SRE calculation**.
        It prints detailed **step-by-step progress** for every chunk ($k$-gap, steps, violations, purities, and SRE values) and automatically **packs completed `.pkl` files into a combined archive after each gap**.
        """
    )
    return


@app.cell
def _(jax, mo):
    # Detect GPU Hardware Status via JAX
    devices = jax.devices("gpu")
    gpu_name = devices[0].device_kind if len(devices) > 0 else "NVIDIA GPU"
    
    gpu_status_md = mo.md(
        f"""
        > [!NOTE]
        > **JAX GPU Accelerator Detected:** `{gpu_name}`  
        > **CUDA 13 Environment:** `cupy-cuda13x`, `jax[cuda13]`, `jaxopt[cuda13]`  
        > **Engine:** Hildebrand Spectral Condition Minimization + In-VRAM FWHT SRE Contraction.
        """
    )
    gpu_status_md
    return devices, gpu_name, gpu_status_md


@app.cell
def _(mo):
    n_qubits_slider = mo.ui.slider(start=4, stop=12, value=7, step=1, label="Qubits (N)")
    num_starts_input = mo.ui.number(start=100, stop=10000, value=2000, step=100, label="Number of Seeds")
    num_steps_input = mo.ui.number(start=100, stop=5000, value=1500, step=100, label="Optimization Steps")
    chunk_size_slider = mo.ui.slider(start=10, stop=250, value=50, step=10, label="Chunk Size (Steps)")
    out_dir_input = mo.ui.text(value="zip2", label="Output Directory")
    run_button = mo.ui.run_button(label="⚡ Start GPU Trajectory Generation")

    control_panel = mo.vstack([
        mo.md("### ⚙️ Simulation Control Panel"),
        mo.hstack([n_qubits_slider, num_starts_input, num_steps_input]),
        mo.hstack([chunk_size_slider, out_dir_input, run_button]),
    ])
    control_panel
    return (
        chunk_size_slider,
        control_panel,
        n_qubits_slider,
        num_starts_input,
        num_steps_input,
        out_dir_input,
        run_button,
    )


@app.cell
def _(os, pickle):
    def pack_pkl_files(source_dir_or_files, output_archive_path):
        if isinstance(source_dir_or_files, str):
            if not os.path.exists(source_dir_or_files):
                print(f"Error: Directory '{source_dir_or_files}' does not exist.")
                return
            file_paths = [
                os.path.join(source_dir_or_files, f)
                for f in sorted(os.listdir(source_dir_or_files))
                if f.endswith(".pkl") and not f.startswith("packed_")
            ]
        else:
            file_paths = list(source_dir_or_files)

        packed_data = {}
        for fpath in file_paths:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "rb") as _f:
                    content = pickle.load(_f)
                packed_data[fname] = content
                print(f"Packed: {fname}")
            except Exception as e:
                print(f"Error reading {fname}: {e}")

        out_dir = os.path.dirname(output_archive_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        try:
            with open(output_archive_path, "wb") as _f:
                pickle.dump(packed_data, _f)
            print(f"\nSuccessfully packed {len(packed_data)} files into archive: {output_archive_path}")
        except Exception as e:
            print(f"Error saving archive: {e}")

    return (pack_pkl_files,)


@app.cell
def _(np):
    # Symplectic Random Generator & State Generator
    def generate_random_generators_symplectic(n_qubits: int, depth_multiplier: int = 10) -> list[str]:
        x_mat = np.zeros((n_qubits, n_qubits), dtype=int)
        z_mat = np.eye(n_qubits, dtype=int)
        r = np.zeros(n_qubits, dtype=int)

        def apply_H(target):
            r[:] ^= x_mat[:, target] & z_mat[:, target]
            x_mat[:, target], z_mat[:, target] = (
                z_mat[:, target].copy(),
                x_mat[:, target].copy(),
            )

        def apply_S(target):
            r[:] ^= x_mat[:, target] & z_mat[:, target]
            z_mat[:, target] ^= x_mat[:, target]

        def apply_CNOT(control, target):
            r[:] ^= (x_mat[:, control] & z_mat[:, target]) & (
                x_mat[:, target] ^ z_mat[:, control] ^ 1
            )
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
                if x == 1 and z == 0:
                    pauli_str += "X"
                elif x == 1 and z == 1:
                    pauli_str += "Y"
                elif x == 0 and z == 1:
                    pauli_str += "Z"
                else:
                    pauli_str += "I"
            generators.append(pauli_str)

        return generators

    PAULI_MAP_NP = {
        "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }

    def pauli_string_to_matrix_np(pauli_str):
        sign = -1 if pauli_str[0] == "-" else 1
        clean_str = pauli_str.lstrip("+-")
        matrices = [PAULI_MAP_NP[char] for char in clean_str]
        mat = matrices[0]
        for m in matrices[1:]:
            mat = np.kron(mat, m)
        return sign * mat

    def build_projector_from_generators_np(generators):
        n_qubits = len(generators[0].lstrip("+-"))
        dim = 2**n_qubits
        projector = np.eye(dim, dtype=np.complex128)
        identity = np.eye(dim, dtype=np.complex128)
        for gen in generators:
            g_matrix = pauli_string_to_matrix_np(gen)
            p_g = (identity + g_matrix) / 2.0
            projector = projector @ p_g
        return projector

    def haar_random_unitary_np(dim: int) -> np.ndarray:
        z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2.0)
        q, r = np.linalg.qr(z)
        d = np.diagonal(r)
        ph = d / np.abs(d)
        return q * ph

    def rand_Almost_Stab_state_np(n_qubits: int, almost_gap: int = 1) -> np.ndarray:
        dim = 2**n_qubits
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = 1.0
        psi = haar_random_unitary_np(dim) @ psi

        if n_qubits - almost_gap == 0:
            return psi

        proj = np.kron(
            build_projector_from_generators_np(
                generate_random_generators_symplectic(n_qubits - almost_gap)
            ),
            np.eye(2**almost_gap, dtype=np.complex128),
        )

        projected_psi = proj @ psi
        norm = np.linalg.norm(projected_psi)
        if norm < 1e-12:
            return psi
        return projected_psi / norm

    return rand_Almost_Stab_state_np, generate_random_generators_symplectic


@app.cell
def _(hadamard, jax, jnp, np):
    # Pure JAX FWHT SRE Functions
    def create_hadamard_jax(n_qubits: int) -> jnp.ndarray:
        dim = 2**n_qubits
        H_np = hadamard(dim).astype(np.float64)
        return jnp.array(H_np)

    def get_pauli_y_phases_jax(n_qubits: int) -> jnp.ndarray:
        dim = 2**n_qubits
        z_grid = np.arange(dim, dtype=np.int32)[:, None]
        x_grid = np.arange(dim, dtype=np.int32)[None, :]
        zx_and = z_grid & x_grid
        num_y_np = np.vectorize(lambda v: bin(v).count('1'))(zx_and)
        phases_np = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)[num_y_np % 4]
        return jnp.array(phases_np)

    def get_xor_indices_jax(n_qubits: int) -> jnp.ndarray:
        dim = 2**n_qubits
        x_idx = np.arange(dim, dtype=np.int32)[:, None]
        b_idx = np.arange(dim, dtype=np.int32)[None, :]
        return jnp.array(x_idx ^ b_idx)

    @jax.jit
    def _compute_sre_sub_batch_jax(psi_sub_batch, H_jax, phases_jax, xor_indices_jax):
        norms = jnp.linalg.norm(psi_sub_batch, axis=1, keepdims=True)
        norms = jnp.where(norms > 1e-12, norms, 1.0)
        psi_normed = psi_sub_batch / norms
        
        psi_conj = jnp.conj(psi_normed)[:, :, None]
        psi_xor = psi_normed[:, xor_indices_jax]
        V_complex = psi_conj * psi_xor
        
        Xi_complex = jnp.matmul(H_jax, V_complex)
        expval_pauli = jnp.real(phases_jax[None, :, :] * Xi_complex)
        
        pauli_4_sum = jnp.sum(expval_pauli ** 4, axis=(1, 2))
        dim = H_jax.shape[0]
        sre_sub = -jnp.log2(pauli_4_sum / dim)
        return sre_sub

    def compute_sre_pure_jax_batch(psi_batch_np: np.ndarray, H_jax, phases_jax, xor_indices_jax, sub_batch_size: int = 250) -> np.ndarray:
        num_starts = psi_batch_np.shape[0]
        results = []
        for i in range(0, num_starts, sub_batch_size):
            sub_batch_jax = jnp.array(psi_batch_np[i : i + sub_batch_size])
            sre_sub = _compute_sre_sub_batch_jax(sub_batch_jax, H_jax, phases_jax, xor_indices_jax)
            results.append(np.array(sre_sub))
        return np.concatenate(results)

    return (
        compute_sre_pure_jax_batch,
        create_hadamard_jax,
        get_pauli_y_phases_jax,
        get_xor_indices_jax,
    )


@app.cell
def _(
    chunk_size_slider,
    combinations,
    compute_sre_pure_jax_batch,
    create_hadamard_jax,
    get_pauli_y_phases_jax,
    get_xor_indices_jax,
    jax,
    jaxopt,
    jnp,
    mo,
    n_qubits_slider,
    np,
    num_starts_input,
    num_steps_input,
    os,
    out_dir_input,
    pack_pkl_files,
    pickle,
    rand_Almost_Stab_state_np,
    run_button,
    time,
    vmap,
):
    # Execution cell: triggers when run_button is clicked with step-by-step progress logging & auto-packing
    if not run_button.value:
        execution_status = mo.md("💡 *Click 'Start GPU Trajectory Generation' above to launch simulation.*")
        completed_files = []
    else:
        n_qubits = n_qubits_slider.value
        num_starts = num_starts_input.value
        num_steps = num_steps_input.value
        chunk_size = chunk_size_slider.value
        out_dir = out_dir_input.value
        sub_batch_size = 250
        
        num_chunks = num_steps // chunk_size
        n_dim = 2**n_qubits
        k = n_qubits // 2
        k_dim = 2**k
        
        combos = list(combinations(range(n_qubits), k))
        perms_list = []
        for combo in combos:
            keep = list(combo)
            trace = [i for i in range(n_qubits) if i not in keep]
            perms_list.append(tuple(trace + keep))

        os.makedirs(out_dir, exist_ok=True)
        out_prefix = os.path.join(out_dir, f"{n_qubits}_qbt_{num_starts}_sds_ptmzng_jfr_")
        archive_path = os.path.join(out_dir, f"packed_{n_qubits}_qbt_{num_starts}_sds_{num_steps}_stps.pkl")
        
        H_jax = create_hadamard_jax(n_qubits)
        phases_jax = get_pauli_y_phases_jax(n_qubits)
        xor_indices_jax = get_xor_indices_jax(n_qubits)
        
        @jax.jit
        def get_purity_and_violation(psi_vec):
            psi = psi_vec[:n_dim] + 1j * psi_vec[n_dim:]
            psi /= jnp.linalg.norm(psi)
            psi_tensor = psi.reshape((2,) * n_qubits)

            rhos = [
                psi_tensor.transpose(perm).reshape(-1, k_dim).conj().T
                @ psi_tensor.transpose(perm).reshape(-1, k_dim)
                for perm in perms_list
            ]
            batch_rho = jnp.stack(rhos, axis=0)

            purities = jnp.sum(jnp.abs(batch_rho)**2, axis=(-2, -1))
            avg_purity = jnp.mean(purities)
            max_purity = jnp.max(purities)

            ex = jnp.linalg.eigvalsh(batch_rho)
            rhs = ex[:, 1] + 2 * jnp.sqrt(jnp.maximum(ex[:, 0] * ex[:, 2], 1e-15))
            viols = jnp.maximum(0.0, ex[:, -1] - rhs)
            total_violation = jnp.sum(viols**2)

            return avg_purity, max_purity, total_violation

        @jax.jit
        def objective_fn(params):
            _, _, total_viol = get_purity_and_violation(params)
            return total_viol

        def single_opt(p):
            solver = jaxopt.LBFGS(
                fun=objective_fn,
                maxiter=chunk_size,
                tol=1e-11,
                history_size=5
            )
            return solver.run(p).params

        vmapped_run = jax.jit(vmap(single_opt))
        vmapped_metrics = jax.jit(vmap(get_purity_and_violation))

        completed_files = []
        status_logs = [f"🚀 **Initialized Restored Hildebrand GPU Run:** `{n_qubits} Qubits` | `{num_starts} Seeds` | `{num_steps} Steps`\n"]
        print(f"=======================================================")
        print(f"GPU DATA GENERATION: {n_qubits} Qubits | {num_starts} Seeds | {num_steps} Steps")
        print(f"=======================================================")

        for k_gap in range(n_qubits, -1, -1):
            gap_header = f"\n⚡ **Processing Gap {k_gap}** (Initial random {k_gap}-qubit stabilized state)..."
            status_logs.append(gap_header)
            print(gap_header)
            
            init_states = [rand_Almost_Stab_state_np(n_qubits, k_gap) for _ in range(num_starts)]
            params_batch_np = np.array([np.concatenate([np.real(s), np.imag(s)]) for s in init_states])

            t0_gap = time.time()
            
            avg_p_list, max_p_list, viol_list = [], [], []
            for i in range(0, num_starts, sub_batch_size):
                p_sub = jnp.array(params_batch_np[i : i + sub_batch_size])
                avg_p, max_p, viol = vmapped_metrics(p_sub)
                avg_p_list.append(np.array(avg_p))
                max_p_list.append(np.array(max_p))
                viol_list.append(np.array(viol))

            avg_p = np.concatenate(avg_p_list)
            max_p = np.concatenate(max_p_list)
            viol = np.concatenate(viol_list)

            states_complex_np = params_batch_np[:, :n_dim] + 1j * params_batch_np[:, n_dim:]
            initial_sre = compute_sre_pure_jax_batch(states_complex_np, H_jax, phases_jax, xor_indices_jax, sub_batch_size=sub_batch_size)

            step0_msg = f"  `Start Step 0` | Violation: `{float(np.mean(viol)):.2e}` | Mean SRE: `{np.mean(initial_sre):.4f}` | Mean Purity: `{float(np.mean(avg_p)):.4f}`"
            status_logs.append(step0_msg)
            print(step0_msg)

            purities_history = [avg_p]
            max_purities_history = [max_p]
            violations_history = [viol]
            sre_history = [initial_sre]

            initial_states_np = np.array(states_complex_np)

            for chunk_idx in range(1, num_chunks + 1):
                new_params_np = np.empty_like(params_batch_np)
                avg_p_list, max_p_list, viol_list = [], [], []

                for i in range(0, num_starts, sub_batch_size):
                    p_sub = jnp.array(params_batch_np[i : i + sub_batch_size])
                    p_opt = vmapped_run(p_sub)
                    avg_p, max_p, viol = vmapped_metrics(p_opt)
                    
                    new_params_np[i : i + sub_batch_size] = np.array(p_opt)
                    avg_p_list.append(np.array(avg_p))
                    max_p_list.append(np.array(max_p))
                    viol_list.append(np.array(viol))

                params_batch_np = new_params_np
                avg_p = np.concatenate(avg_p_list)
                max_p = np.concatenate(max_p_list)
                viol = np.concatenate(viol_list)

                purities_history.append(avg_p)
                max_purities_history.append(max_p)
                violations_history.append(viol)

                states_complex_np = params_batch_np[:, :n_dim] + 1j * params_batch_np[:, n_dim:]
                sre_vals = compute_sre_pure_jax_batch(states_complex_np, H_jax, phases_jax, xor_indices_jax, sub_batch_size=sub_batch_size)
                sre_history.append(sre_vals)

                step_num = chunk_idx * chunk_size
                elapsed = time.time() - t0_gap
                est_rem = (elapsed / chunk_idx) * (num_chunks - chunk_idx)
                
                step_msg = f"  `Step {step_num:5d}/{num_steps}` | Violation: `{float(np.mean(viol)):.2e}` | Mean SRE: `{np.mean(sre_vals):.4f}` | Elapsed: `{elapsed:.1f}s` | Est. Rem: `{est_rem:.1f}s`"
                status_logs.append(step_msg)
                print(step_msg)

            final_states_np = np.array(states_complex_np)

            purities_history = np.array(purities_history).T.tolist()
            max_purities_history = np.array(max_purities_history).T.tolist()
            violations_history = np.array(violations_history).T.tolist()
            sre_history = np.array(sre_history).T.tolist()

            save_filepath = f"{out_prefix}{num_steps}_stps_{k_gap}.pkl"
            results_dict = {
                'initial_states': initial_states_np,
                'final_states': final_states_np,
                'average_purity': purities_history,
                'max_purity': max_purities_history,
                'total_violation': violations_history,
                'sre': sre_history,
                'opt_history': []
            }

            with open(save_filepath, 'wb') as _f_save:
                pickle.dump(results_dict, _f_save)

            completed_files.append(save_filepath)
            saved_msg = f"  ✅ **Gap {k_gap} Saved:** `{save_filepath}` (Total Gap Time: `{time.time() - t0_gap:.1f}s`)"
            status_logs.append(saved_msg)
            print(saved_msg)

            # Auto-pack files after each gap
            pack_pkl_files(completed_files, archive_path)
            pack_msg = f"  📦 **Packed Archive Updated:** `{archive_path}` ({len(completed_files)} files)"
            status_logs.append(pack_msg)
            print(pack_msg)

        execution_status = mo.md("\n\n".join(status_logs))
    
    execution_status
    return completed_files, execution_status


@app.cell
def _(completed_files, go, make_subplots, mo, np, pickle):
    # Interactive Trajectory Visualization Cell
    if not completed_files:
        plot_display = mo.md("📈 *Completed trajectory plots will appear here once simulation starts.*")
    else:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Mean SRE (S₂)", "Mean Purity (Tr ρ²)", "Mean Constraint Violation")
        )
        
        for file in completed_files:
            with open(file, 'rb') as _f_load:
                data = pickle.load(_f_load)
            
            sre_mean = np.mean(data['sre'], axis=0)
            purity_mean = np.mean(data['average_purity'], axis=0)
            viol_mean = np.mean(data['total_violation'], axis=0)
            steps = np.arange(len(sre_mean)) * 50

            gap_label = file.split("_")[-1].replace(".pkl", "")
            
            fig.add_trace(go.Scatter(x=steps, y=sre_mean, mode='lines', name=f"Gap {gap_label} SRE"), row=1, col=1)
            fig.add_trace(go.Scatter(x=steps, y=purity_mean, mode='lines', name=f"Gap {gap_label} Purity"), row=1, col=2)
            fig.add_trace(go.Scatter(x=steps, y=viol_mean, mode='lines', name=f"Gap {gap_label} Violation"), row=1, col=3)

        fig.update_layout(height=450, title_text="⚡ Live GPU Optimization Trajectories", template="plotly_dark")
        plot_display = mo.ui.plotly(fig)
        
    plot_display
    return fig, plot_display


if __name__ == "__main__":
    app.run()
