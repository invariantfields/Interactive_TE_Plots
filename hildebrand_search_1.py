# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "jax[cuda12]",
#     "jaxopt",
#     "plotly",
#     "scipy",
#     "cupy-cuda12x",
#     "juliacall",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    import site

    # Fix for JAX: Point to pip-installed CUDA libraries if they exist
    # This prevents JAX from falling back to CPU due to outdated system cuBLAS
    try:
        # Check current environment's site-packages
        _site_packages = site.getsitepackages()[0]
        _nvidia_base = os.path.join(_site_packages, "nvidia")
        if os.path.exists(_nvidia_base):
            _library_paths = []
            for _subdir in os.listdir(_nvidia_base):
                _lib_dir = os.path.join(_nvidia_base, _subdir, "lib")
                if os.path.exists(_lib_dir):
                    _library_paths.append(_lib_dir)

            # Add NVVM if present
            _nvvm_dir = os.path.join(_nvidia_base, "cuda_nvcc", "nvvm", "lib64")
            if os.path.exists(_nvvm_dir):
                _library_paths.append(_nvvm_dir)

            _current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            _new_ld = ":".join(_library_paths + ([_current_ld] if _current_ld else []))
            os.environ["LD_LIBRARY_PATH"] = _new_ld
    except Exception:
        pass

    sys.setrecursionlimit(10000)

    from juliacall import Main as jl

    # Suppress implementation warnings (e.g. progress bars)
    jl.seval("using Logging; disable_logging(Logging.Warn)")
    jl.seval("using HadaMAG")
    jl.seval("using CUDA")

    import numpy as np
    import jax
    import jax.numpy as jnp
    import cupy as cp
    from functools import reduce
    from jaxopt import LBFGS
    import plotly.graph_objects as go
    import pickle
    from itertools import combinations
    import time
    import traceback

    # Enable 64-bit precision for JAX
    jax.config.update("jax_enable_x64", True)

    # Print device info for diagnostics
    print(f"JAX Devices: {jax.devices()}")
    return (
        LBFGS,
        combinations,
        cp,
        go,
        jax,
        jl,
        jnp,
        mo,
        np,
        os,
        pickle,
        reduce,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Optimized Hildebrand Search (JAX Version)
    This notebook uses **JAX** and **Auto-Grad** to find quantum states satisfying the **1-vs-rest ASEP condition**.
    It also includes **Julia/HadaMAG** for SRE computations and **CuPy** for Stabilizer state initialization.
    """)
    return


@app.cell
def _(cp, jl, np):
    # =====================================================================
    # Utilities: SRE, Partial Trace, is_appt, and Hildebrand
    # =====================================================================
    def compute_sre(psi_np, alpha=2):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dim = len(psi_np)
                n_qubits = int(np.log2(dim))

                jl.psi_python = np.array(psi_np)
                jl.alpha = alpha
                jl.n_qubits = n_qubits
                jl.dim = dim

                jl.seval(""" 
                    psi_jl = Vector{ComplexF64}(psi_python)
                    psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, n_qubits, dim)
                    sre_result, lost_norm = SRE(psi_sv, alpha, backend= :CUDA)
                """)

                return jl.sre_result, jl.lost_norm
            except Exception as e:
                print(f"Execution Error: {e}")
                return None, None

    def par_trace(psi, dim, n, n_parties):
        """Optimized partial trace for pure states."""
        n_rem = n - n_parties
        psi_mat = psi.reshape(dim**n_rem, dim**n_parties)
        return psi_mat @ psi_mat.conj().T

    def is_appt(x: cp.ndarray) -> bool:
        _purity = cp.sum(x * x)
        _D = cp.shape(x)[0]
        if _purity <= 1 / (_D - 1):
            return True
        _ex, _ = cp.linalg.eigh(x)
        _ex = cp.asnumpy(_ex)
        if _ex[-1] <= _ex[1] + 2 * np.sqrt(_ex[0] * _ex[2]):
            return True
        return False

    def get_hildebrand_violation(rho, k_dim):
        """Calculates violation of the 1-vs-rest ASEP condition on GPU."""
        ex = cp.linalg.eigvalsh(rho)
        rhs = ex[1] + 2 * cp.sqrt(cp.maximum(ex[0] * ex[2], 0))
        violation = cp.maximum(0, ex[-1] - rhs)
        return float(violation)

    return compute_sre, get_hildebrand_violation, is_appt, par_trace


@app.cell
def _(cp, np, reduce):
    # =====================================================================
    # Initial State Generation (Random Stabilizers)
    # =====================================================================
    def generate_random_generators_symplectic(
        n_qubits: int, depth_multiplier: int = 10
    ) -> list[str]:
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
            elif n_qubits > 1:
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

    PAULI_MAP = {
        "I": cp.array([[1, 0], [0, 1]], dtype=complex),
        "X": cp.array([[0, 1], [1, 0]], dtype=complex),
        "Y": cp.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": cp.array([[1, 0], [0, -1]], dtype=complex),
    }

    def pauli_string_to_matrix(pauli_str):
        sign = -1 if pauli_str[0] == "-" else 1
        matrices = [PAULI_MAP[char] for char in pauli_str.lstrip("+-")]
        return sign * reduce(cp.kron, matrices)

    def build_projector_from_generators(generators):
        n_qubits = len(generators[0].lstrip("+-"))
        dim = 2**n_qubits
        projector = cp.eye(dim, dtype=complex)
        identity = cp.eye(dim, dtype=complex)
        for gen in generators:
            p_g = (identity + pauli_string_to_matrix(gen)) / 2.0
            projector = projector @ p_g
        return projector

    def haar_random_unitary_gpu(dim: int) -> cp.ndarray:
        z = cp.random.randn(dim, dim) + 1j * cp.random.randn(dim, dim)
        q, r = cp.linalg.qr(z)
        d = cp.diagonal(r)
        return q * (d / cp.abs(d))

    def rand_Almost_Stab_state(n_qubits: int, almost_gap: int = 1) -> cp.ndarray:
        psi = cp.zeros(2**n_qubits, dtype=complex)
        psi[0] = 1.0
        psi = haar_random_unitary_gpu(2**n_qubits) @ psi

        if n_qubits - almost_gap <= 0:
            return psi

        proj = cp.kron(
            build_projector_from_generators(
                generate_random_generators_symplectic(n_qubits - almost_gap)
            ),
            cp.eye(2**almost_gap, dtype=complex),
        )
        projected_psi = proj @ psi
        return projected_psi / cp.linalg.norm(projected_psi)

    return (rand_Almost_Stab_state,)


@app.cell
def _(combinations, jax, jnp):
    # =====================================================================
    # JAX Optimization Logic
    # =====================================================================
    def get_search_logic(n, k):
        """Generates JIT-compiled search logic for a specific N, K."""
        combos = list(combinations(range(n), k))

        perms = []
        for combo in combos:
            keep = list(combo)
            trace = [i for i in range(n) if i not in keep]
            perms.append(tuple(trace + keep))

        @jax.jit
        def get_metrics(params):
            """Calculates purity and violation for all marginals."""
            n_dim = 2**n
            psi = params[:n_dim] + 1j * params[n_dim:]
            psi /= jnp.linalg.norm(psi)
            psi_tensor = psi.reshape((2,) * n)

            avg_purity = 0.0
            total_violation = 0.0

            for perm in perms:
                psi_perm = psi_tensor.transpose(perm).reshape(-1, 2**k)
                rho = psi_perm.conj().T @ psi_perm

                avg_purity += jnp.real(jnp.sum(rho * rho))

                ex = jnp.linalg.eigvalsh(rho)
                # Improved regularization for gradient flow at zero eigenvalues
                rhs = ex[1] + 2 * jnp.sqrt(ex[0] * ex[2] + 1e-15)
                viol = jnp.maximum(0, ex[-1] - rhs)
                total_violation += viol**2

            return avg_purity / len(perms), total_violation

        return get_metrics, combos

    return (get_search_logic,)


@app.cell
def _(mo):
    n_qubits_slider = mo.ui.number(
        3, 10, step=1, value=4, label="Number of Qubits (N)"
    )
    max_iter_input = mo.ui.number(
        10, 10000, step=10, value=200, label="Max Iterations"
    )
    run_button = mo.ui.run_button(label="🚀 Start Single Search")

    # NEW: Batch UI
    n_samples_input = mo.ui.number(
        1, 100, step=1, value=5, label="Samples per Gap"
    )
    run_batch_button = mo.ui.run_button(label="📊 Start Batch Analysis (All Gaps)")
    return (
        max_iter_input,
        n_qubits_slider,
        n_samples_input,
        run_batch_button,
        run_button,
    )


@app.cell
def _(mo, n_qubits_slider):
    gap_slider = mo.ui.number(
        0, n_qubits_slider.value, step=1, value=1, label="Almost Stabilizer Gap"
    )
    return (gap_slider,)


@app.cell
def _(
    gap_slider,
    max_iter_input,
    mo,
    n_qubits_slider,
    n_samples_input,
    run_batch_button,
    run_button,
):
    single_col = mo.vstack(
        [
            mo.md("#### Single State Search"),
            mo.hstack([gap_slider, run_button], gap=1),
        ]
    )

    batch_col = mo.vstack(
        [
            mo.md("#### Batch Analysis"),
            mo.hstack([n_samples_input, run_batch_button], gap=1),
        ]
    )

    layout = mo.vstack(
        [
            mo.md("### Search Parameters"),
            mo.hstack(
                [n_qubits_slider, max_iter_input],
                justify="start",
                gap=2,
            ),
            mo.hstack([single_col, batch_col], gap=4, justify="start"),
        ]
    )
    layout
    return


@app.cell
def _(
    LBFGS,
    compute_sre,
    cp,
    gap_slider,
    get_hildebrand_violation,
    get_search_logic,
    is_appt,
    jnp,
    max_iter_input,
    mo,
    n_qubits_slider,
    np,
    par_trace,
    rand_Almost_Stab_state,
    run_button,
):
    mo.stop(not run_button.value)

    _n = n_qubits_slider.value
    _gap = gap_slider.value
    _k = _n // 2
    _n_dim = 2**_n

    with mo.status.spinner(title=f"Initializing random state (Gap: {_gap})..."):
        init_psi_cp = rand_Almost_Stab_state(_n, _gap)
        init_psi_np = cp.asnumpy(init_psi_cp)
        # Add tiny noise to break stabilizer symmetry (prevents zero-gradient trap)
        _noise = np.random.normal(size=init_psi_np.shape) + 1j * np.random.normal(
            size=init_psi_np.shape
        )
        init_psi_np = init_psi_np + 1e-6 * _noise
        init_psi_np /= np.linalg.norm(init_psi_np)

    _get_metrics, _combos = get_search_logic(_n, _k)

    # Calculate BEFORE stats
    before_stats = {}
    with mo.status.spinner(title="Calculating initial state metrics..."):
        for _combo in _combos:
            _per = list(set(range(_n)) - set(_combo)) + list(_combo)
            _psi_moved = cp.moveaxis(
                init_psi_cp.reshape([2] * _n), list(range(_n)), _per
            ).flatten()
            _rho_cp = par_trace(_psi_moved, 2, _n, _k)
            before_stats[str(_combo)] = bool(is_appt(_rho_cp))
            del _psi_moved, _rho_cp

    with mo.status.spinner(title="Computing SRE on initial state..."):
        init_sre_val, _ = compute_sre(init_psi_np)

    def _objective(params):
        _, total_viol = _get_metrics(params)
        return total_viol

    _init_params = jnp.concatenate(
        [jnp.real(init_psi_np), jnp.imag(init_psi_np)]
    ).astype(jnp.float64)
    _solver = LBFGS(fun=_objective, maxiter=max_iter_input.value, tol=1e-12)

    with mo.status.spinner(title=f"Optimizing {_n}-qubit state using JAX..."):
        _res = _solver.run(_init_params)

    _final_params = _res.params
    final_psi_np = np.array(
        _final_params[:_n_dim] + 1j * _final_params[_n_dim:]
    )
    final_psi_np /= np.linalg.norm(final_psi_np)

    avg_p, total_v = _get_metrics(_final_params)

    with mo.status.spinner(title="Computing SRE (Julia)..."):
        sre_val, _ = compute_sre(final_psi_np)

    # Calculate AFTER stats
    results = []
    _final_psi_cp = cp.asarray(final_psi_np)

    with mo.status.spinner(title="Generating verification report..."):
        for _combo in _combos:
            _per = list(set(range(_n)) - set(_combo)) + list(_combo)
            _psi_moved = cp.moveaxis(
                _final_psi_cp.reshape([2] * _n), list(range(_n)), _per
            ).flatten()
            _rho_cp = par_trace(_psi_moved, 2, _n, _k)

            _viol = get_hildebrand_violation(_rho_cp, _k)
            _purity = float(cp.sum(_rho_cp * _rho_cp).real)
            _linear_entropy = 1.0 - _purity
            appt_after = bool(is_appt(_rho_cp))

            results.append(
                {
                    "Qubits": str(_combo),
                    "is_appt (Before)": "TRUE"
                    if before_stats[str(_combo)]
                    else "FALSE",
                    "is_appt (After)": "TRUE" if appt_after else "FALSE",
                    "1 - Purity": f"{_linear_entropy:.5f}",
                    "Violation": f"{_viol:.2e}",
                    "Status": "✅ PASS" if _viol < 1e-8 else "❌ FAIL",
                }
            )
            del _psi_moved, _rho_cp

    status_msg = mo.md(
        f"**Optimization Complete**\n- Final Violation Score: `{_res.state.error:.2e}`\n- SRE: `{sre_val}`"
    )
    return avg_p, init_sre_val, results, sre_val, status_msg, total_v


@app.cell
def _(avg_p, init_sre_val, mo, results, sre_val, status_msg, total_v):
    mo.stop(not results)

    report = mo.vstack(
        [
            status_msg,
            mo.md("### Verification Report"),
            mo.md(
                f"**Average Purity:** `{float(avg_p):.5f}` | **Initial SRE:** `{init_sre_val}` | **Final SRE:** `{sre_val}` | **Total Violation:** `{float(total_v):.2e}`"
            ),
            mo.ui.table(results),
        ]
    )
    report
    return


@app.cell
def _(
    LBFGS,
    compute_sre,
    cp,
    get_search_logic,
    go,
    jnp,
    max_iter_input,
    mo,
    n_qubits_slider,
    n_samples_input,
    np,
    os,
    pickle,
    rand_Almost_Stab_state,
    run_batch_button,
):
    mo.stop(not run_batch_button.value)

    _n_batch = n_qubits_slider.value
    _n_samples = n_samples_input.value
    _k_sub = _n_batch // 2
    _n_dim_batch = 2**_n_batch

    _get_metrics_batch, _ = get_search_logic(_n_batch, _k_sub)

    def _objective_batch(params):
        _, total_viol = _get_metrics_batch(params)
        return total_viol

    _solver_batch = LBFGS(fun=_objective_batch, maxiter=max_iter_input.value, tol=1e-12)

    _gaps = list(range(_n_batch + 1))
    _means_initial, _stds_initial = [], []
    _means_final, _stds_final = [], []
    _means_filtered, _stds_filtered = [], []

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    for _gap_val in _gaps:
        _init_sres = []
        _final_sres = []
        _filtered_sres = []
        _final_states = []
        _violation_scores = []

        with mo.status.spinner(
            title=f"Processing Gap {_gap_val}/{_n_batch} ({_n_samples} samples)..."
        ):
            for _s_idx in range(_n_samples):
                # 1. Initialize with symmetry breaking
                _psi_cp_b = rand_Almost_Stab_state(_n_batch, _gap_val)
                _psi_np_b = cp.asnumpy(_psi_cp_b)
                _noise_b = np.random.normal(
                    size=_psi_np_b.shape
                ) + 1j * np.random.normal(size=_psi_np_b.shape)
                _psi_np_b = _psi_np_b + 1e-6 * _noise_b
                _psi_np_b /= np.linalg.norm(_psi_np_b)

                # 2. Initial SRE
                _sre_init_b, _ = compute_sre(_psi_np_b)
                if _sre_init_b is not None:
                    _init_sres.append(float(_sre_init_b))

                # 3. Optimize
                _init_params_b = jnp.concatenate(
                    [jnp.real(_psi_np_b), jnp.imag(_psi_np_b)]
                )
                _res_b = _solver_batch.run(_init_params_b)

                # 4. Final State
                _final_params_b = _res_b.params
                _final_psi_b = np.array(
                    _final_params_b[:_n_dim_batch]
                    + 1j * _final_params_b[_n_dim_batch:]
                )
                _final_psi_b /= np.linalg.norm(_final_psi_b)
                _final_states.append(_final_psi_b)

                # 5. Final Metrics
                _sre_final_b, _ = compute_sre(_final_psi_b)
                _, _viol_score_b = _get_metrics_batch(_final_params_b)
                _violation_scores.append(float(_viol_score_b))

                if _sre_final_b is not None:
                    _final_sres.append(float(_sre_final_b))
                    if _viol_score_b < 1e-8:
                        _filtered_sres.append(float(_sre_final_b))

        # Save raw data for this gap
        _batch_data = {
            "n_qubits": _n_batch,
            "gap": _gap_val,
            "n_samples": _n_samples,
            "initial_sre": _init_sres,
            "final_sre": _final_sres,
            "filtered_sre": _filtered_sres,
            "violation_scores": _violation_scores,
            "final_states": _final_states,
        }
        _filename = f"data/hildebrand_batch_{_n_batch}q_{_n_samples}s_gap{_gap_val}.pkl"
        with open(_filename, "wb") as f:
            pickle.dump(_batch_data, f)

        _means_initial.append(np.mean(_init_sres) if _init_sres else 0)
        _stds_initial.append(np.std(_init_sres) if _init_sres else 0)
        _means_final.append(np.mean(_final_sres) if _final_sres else 0)
        _stds_final.append(np.std(_final_sres) if _final_sres else 0)
        _means_filtered.append(np.mean(_filtered_sres) if _filtered_sres else 0)
        _stds_filtered.append(np.std(_filtered_sres) if _filtered_sres else 0)

    # Build Chart
    _fig_batch = go.Figure()
    _fig_batch.add_trace(
        go.Bar(
            name="Initial SRE",
            x=_gaps,
            y=_means_initial,
            error_y=dict(type="data", array=_stds_initial),
            marker_color="indianred",
        )
    )
    _fig_batch.add_trace(
        go.Bar(
            name="Final SRE",
            x=_gaps,
            y=_means_final,
            error_y=dict(type="data", array=_stds_final),
            marker_color="lightsalmon",
        )
    )
    _fig_batch.add_trace(
        go.Bar(
            name="Filtered SRE (ASEP Passed)",
            x=_gaps,
            y=_means_filtered,
            error_y=dict(type="data", array=_stds_filtered),
            marker_color="seagreen",
        )
    )

    _fig_batch.update_layout(
        title=f"SRE Comparison for {_n_batch}-Qubits (k={_k_sub} subspace analysis)",
        xaxis_title="Almost-Stabilizer Gap (k)",
        yaxis_title="Average SRE (α=2)",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    batch_result_view = mo.vstack(
        [mo.md("### Batch Results"), mo.as_html(_fig_batch)]
    )
    batch_result_view
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
