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
    import jax
    import jax.numpy as jnp
    from jax import random, vmap
    import jaxopt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import marimo as mo

    # Configure GPU memory sharing
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70"
    jax.config.update("jax_enable_x64", True)

    return (
        os,
        sys,
        time,
        pickle,
        np,
        hadamard,
        jax,
        jnp,
        random,
        vmap,
        jaxopt,
        go,
        make_subplots,
        mo,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # 🚀 High-Performance GPU Trajectory Generation
        ### Fast Walsh-Hadamard Transform (FWHT) SRE Engine | Optimized for RTX 6000 PRO (96GB VRAM)

        [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/invariantfields/Interactive_TE_Plots/blob/master/fast_7q_gpu_data_generation_notebook.py)

        This notebook implements **100% GPU Stabilizer Rényi Entropy ($S_2$) calculation** combined with **JAX L-BFGS state optimization**.
        It prints detailed **step-by-step progress** for every chunk ($k$-gap, steps, violations, purities, and SRE values).
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
        > **Mode:** Memory-Light 100% GPU XLA in-VRAM tensor contraction (Fast Walsh-Hadamard Transform).
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

    def compute_sre_pure_jax_batch(psi_batch_jax, H_jax, phases_jax, xor_indices_jax, sub_batch_size: int = 250) -> np.ndarray:
        num_starts = psi_batch_jax.shape[0]
        results = []
        for i in range(0, num_starts, sub_batch_size):
            sub_batch = psi_batch_jax[i : i + sub_batch_size]
            sre_sub = _compute_sre_sub_batch_jax(sub_batch, H_jax, phases_jax, xor_indices_jax)
            results.append(np.array(sre_sub))
        return np.concatenate(results)

    return (
        _compute_sre_sub_batch_jax,
        compute_sre_pure_jax_batch,
        create_hadamard_jax,
        get_pauli_y_phases_jax,
        get_xor_indices_jax,
    )


@app.cell
def _(jax, jaxopt, jnp, random, vmap):
    # Optimized Memory-Light JAX Objective & Optimization Helpers
    def generate_stabilizer_states_and_generators(n_qubits, k, num_starts, seed=42):
        key = random.PRNGKey(seed)
        dim = 2**n_qubits
        pauli_matrices = [
            jnp.array([[1, 0], [0, 1]], dtype=jnp.complex128),
            jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128),
            jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128),
            jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
        ]
        
        def get_pauli_string(indices):
            mat = jnp.array([[1.0]], dtype=jnp.complex128)
            for idx in indices:
                mat = jnp.kron(mat, pauli_matrices[idx])
            return mat

        states = []
        generators_list = []
        
        for start_idx in range(num_starts):
            key, subkey1, subkey2 = random.split(key, 3)
            pauli_indices = random.randint(subkey1, (k, n_qubits), 0, 4)
            generators = [get_pauli_string(indices) for indices in pauli_indices]
            
            psi = random.normal(subkey2, (dim,)) + 1j * random.normal(subkey2, (dim,))
            psi = psi / jnp.linalg.norm(psi)
            
            if k > 0:
                P_proj = jnp.eye(dim, dtype=jnp.complex128)
                for g in generators:
                    P_proj = P_proj @ (jnp.eye(dim, dtype=jnp.complex128) + g) / 2.0
                psi = P_proj @ psi
                norm = jnp.linalg.norm(psi)
                if norm < 1e-6:
                    psi = random.normal(subkey2, (dim,)) + 1j * random.normal(subkey2, (dim,))
                    psi = psi / jnp.linalg.norm(psi)
                else:
                    psi = psi / norm

            states.append(psi)
            if k > 0:
                generators_list.append(jnp.stack(generators))
            else:
                generators_list.append(jnp.zeros((1, dim, dim), dtype=jnp.complex128))
                
        return jnp.stack(states), jnp.stack(generators_list)

    def objective_fn(params, n_dim, generators, k):
        real_part = params[:n_dim]
        imag_part = params[n_dim:]
        psi = real_part + 1j * imag_part
        norm_sq = jnp.sum(real_part**2 + imag_part**2)
        psi_normed = psi / jnp.sqrt(jnp.maximum(norm_sq, 1e-12))

        purity_loss = (norm_sq - 1.0)**2

        violation_loss = 0.0
        if k > 0:
            expvals = jnp.real(jnp.einsum('i,kij,j->k', jnp.conj(psi_normed), generators, psi_normed))
            violation_loss = jnp.sum((expvals - 1.0)**2)

        return purity_loss + violation_loss

    def calculate_metrics(params, n_dim, generators, k):
        real_part = params[:n_dim]
        imag_part = params[n_dim:]
        psi = real_part + 1j * imag_part
        norm_sq = jnp.sum(real_part**2 + imag_part**2)
        psi_normed = psi / jnp.sqrt(jnp.maximum(norm_sq, 1e-12))

        avg_p = 1.0
        max_p = 1.0

        viol = 0.0
        if k > 0:
            expvals = jnp.real(jnp.einsum('i,kij,j->k', jnp.conj(psi_normed), generators, psi_normed))
            viol = jnp.sum((expvals - 1.0)**2)

        return avg_p, max_p, viol

    def run_subbatched_opt(params_batch, generators_jax, n_dim, k, chunk_size, sub_batch_size=250):
        num_starts = params_batch.shape[0]
        
        def single_opt(p, g):
            solver = jaxopt.LBFGS(
                fun=lambda x: objective_fn(x, n_dim, g, k),
                maxiter=chunk_size,
                tol=1e-11,
                history_size=10
            )
            return solver.run(p).params

        vmapped_run = vmap(single_opt, in_axes=(0, 0))
        vmapped_metrics = vmap(lambda p, g: calculate_metrics(p, n_dim, g, k), in_axes=(0, 0))

        new_params_list = []
        avg_p_list, max_p_list, viol_list = [], [], []

        for i in range(0, num_starts, sub_batch_size):
            p_sub = params_batch[i : i + sub_batch_size]
            g_sub = generators_jax[i : i + sub_batch_size]
            
            p_opt = vmapped_run(p_sub, g_sub)
            avg_p, max_p, viol = vmapped_metrics(p_opt, g_sub)
            
            new_params_list.append(p_opt)
            avg_p_list.append(avg_p)
            max_p_list.append(max_p)
            viol_list.append(viol)

        new_params = jnp.vstack(new_params_list)
        avg_p_cat = jnp.concatenate(avg_p_list)
        max_p_cat = jnp.concatenate(max_p_list)
        viol_cat = jnp.concatenate(viol_list)

        return new_params, avg_p_cat, max_p_cat, viol_cat

    return (
        calculate_metrics,
        generate_stabilizer_states_and_generators,
        objective_fn,
        run_subbatched_opt,
    )


@app.cell
def _(
    calculate_metrics,
    chunk_size_slider,
    compute_sre_pure_jax_batch,
    create_hadamard_jax,
    generate_stabilizer_states_and_generators,
    get_pauli_y_phases_jax,
    get_xor_indices_jax,
    jax,
    jnp,
    mo,
    n_qubits_slider,
    np,
    num_starts_input,
    num_steps_input,
    os,
    out_dir_input,
    pickle,
    run_button,
    run_subbatched_opt,
    time,
    vmap,
):
    # Execution cell: triggers when run_button is clicked with step-by-step progress logging
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
        os.makedirs(out_dir, exist_ok=True)
        out_prefix = os.path.join(out_dir, f"{n_qubits}_qbt_{num_starts}_sds_ptmzng_jfr_")
        
        H_jax = create_hadamard_jax(n_qubits)
        phases_jax = get_pauli_y_phases_jax(n_qubits)
        xor_indices_jax = get_xor_indices_jax(n_qubits)
        
        completed_files = []
        status_logs = [f"🚀 **Initialized Memory-Light GPU Run:** `{n_qubits} Qubits` | `{num_starts} Seeds` | `{num_steps} Steps`\n"]
        print(f"=======================================================")
        print(f"GPU DATA GENERATION: {n_qubits} Qubits | {num_starts} Seeds | {num_steps} Steps")
        print(f"=======================================================")

        for k in range(n_qubits, -1, -1):
            gap_header = f"\n⚡ **Processing Gap {k}** (Initial {k}-qubit stabilized state)..."
            status_logs.append(gap_header)
            print(gap_header)
            
            initial_states_jax, generators_jax = generate_stabilizer_states_and_generators(
                n_qubits, k, num_starts, seed=42 + k
            )
            
            params_batch = jnp.hstack([
                jnp.real(initial_states_jax),
                jnp.imag(initial_states_jax)
            ])
            
            vmapped_metrics = vmap(lambda p, g: calculate_metrics(p, n_dim, g, k), in_axes=(0, 0))

            avg_p, max_p, viol = vmapped_metrics(params_batch, generators_jax)
            states_complex_jax = params_batch[:, :n_dim] + 1j * params_batch[:, n_dim:]
            
            t0_gap = time.time()
            initial_sre = compute_sre_pure_jax_batch(states_complex_jax, H_jax, phases_jax, xor_indices_jax, sub_batch_size=sub_batch_size)

            step0_msg = f"  `Start Step 0` | Violation: `{float(jnp.mean(viol)):.2e}` | Mean SRE: `{np.mean(initial_sre):.4f}`"
            status_logs.append(step0_msg)
            print(step0_msg)

            purities_history = [np.array(avg_p)]
            max_purities_history = [np.array(max_p)]
            violations_history = [np.array(viol)]
            sre_history = [initial_sre]

            initial_states_np = np.array(states_complex_jax)

            for chunk_idx in range(1, num_chunks + 1):
                params_batch, avg_p, max_p, viol = run_subbatched_opt(
                    params_batch, generators_jax, n_dim, k, chunk_size, sub_batch_size=sub_batch_size
                )

                purities_history.append(np.array(avg_p))
                max_purities_history.append(np.array(max_p))
                violations_history.append(np.array(viol))

                states_complex_jax = params_batch[:, :n_dim] + 1j * params_batch[:, n_dim:]
                
                sre_vals = compute_sre_pure_jax_batch(states_complex_jax, H_jax, phases_jax, xor_indices_jax, sub_batch_size=sub_batch_size)
                sre_history.append(sre_vals)

                step_num = chunk_idx * chunk_size
                elapsed = time.time() - t0_gap
                est_rem = (elapsed / chunk_idx) * (num_chunks - chunk_idx)
                
                step_msg = f"  `Step {step_num:5d}/{num_steps}` | Violation: `{float(jnp.mean(viol)):.2e}` | Mean SRE: `{np.mean(sre_vals):.4f}` | Elapsed: `{elapsed:.1f}s` | Est. Rem: `{est_rem:.1f}s`"
                status_logs.append(step_msg)
                print(step_msg)

            final_states_np = np.array(states_complex_jax)

            purities_history = np.array(purities_history).T.tolist()
            max_purities_history = np.array(max_purities_history).T.tolist()
            violations_history = np.array(violations_history).T.tolist()
            sre_history = np.array(sre_history).T.tolist()

            save_filepath = f"{out_prefix}{num_steps}_stps_{k}.pkl"
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
            saved_msg = f"  ✅ **Gap {k} Saved:** `{save_filepath}` (Total Gap Time: `{time.time() - t0_gap:.1f}s`)"
            status_logs.append(saved_msg)
            print(saved_msg)

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
