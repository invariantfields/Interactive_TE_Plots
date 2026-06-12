import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import sys

    sys.setrecursionlimit(10000)

    from juliacall import Main as jl
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc

    jl.seval("using HadaMAG")
    jl.seval("using CUDA")
    import numpy as np
    import cupy as cp
    from functools import reduce
    from scipy.stats import unitary_group
    import scipy.linalg as la
    from itertools import combinations, repeat
    import pickle
    from statistics import mean

    return (
        combinations,
        cp,
        go,
        jl,
        make_subplots,
        mean,
        mo,
        np,
        pc,
        pickle,
        reduce,
    )


@app.cell
def _(mo):
    # OPTIMIZATION: Created run buttons to prevent automatic execution of heavy tasks
    run_sim_btn = mo.ui.run_button(label="▶ Run Simulation")
    run_diag_btn = mo.ui.run_button(label="▶ Run Diagnostics")

    mo.md(
        f"""
        ### Control Panel
        Use these buttons to trigger expensive computations manually.
        {run_sim_btn} {run_diag_btn}
        """
    )
    return run_diag_btn, run_sim_btn


@app.cell
def _(np):
    # =====================================================================
    # 1. Symplectic Generator (Kept as NumPy / CPU)
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
            elif n_qubits - 2 > 0:
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

    return (generate_random_generators_symplectic,)


@app.cell
def _(cp):
    # =====================================================================
    # 2. Dense Projector Construction (GPU Accelerated)
    # =====================================================================
    PAULI_MAP = {
        "I": cp.array([[1, 0], [0, 1]], dtype=complex),
        "X": cp.array([[0, 1], [1, 0]], dtype=complex),
        "Y": cp.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": cp.array([[1, 0], [0, -1]], dtype=complex),
    }
    return (PAULI_MAP,)


@app.cell
def _(PAULI_MAP, cp, reduce):
    def pauli_string_to_matrix(pauli_str):
        sign = -1 if pauli_str[0] == "-" else 1
        clean_str = pauli_str.lstrip("+-")
        matrices = [PAULI_MAP[char] for char in clean_str]
        return sign * reduce(cp.kron, matrices)

    return (pauli_string_to_matrix,)


@app.cell
def _(cp, pauli_string_to_matrix):
    def build_projector_from_generators(generators):
        n_qubits = len(generators[0].lstrip("+-"))
        dim = 2**n_qubits

        projector = cp.eye(dim, dtype=complex)
        identity = cp.eye(dim, dtype=complex)

        for gen in generators:
            g_matrix = pauli_string_to_matrix(gen)
            p_g = (identity + g_matrix) / 2.0
            projector = projector @ p_g

        return projector

    return (build_projector_from_generators,)


@app.cell
def _(
    build_projector_from_generators,
    cp,
    generate_random_generators_symplectic,
    haar_random_unitary_gpu,
):
    def rand_Almost_Stab_state(n_qubits: int, almost_gap: int = 1) -> cp.ndarray:
        psi = cp.zeros(2**n_qubits, dtype=complex)
        psi[0] = 1.0
        psi = haar_random_unitary_gpu(2**n_qubits) @ psi

        if n_qubits - almost_gap == 0:
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
def _(cp, haar_random_unitary_gpu):
    def haar_random_state(n_qubits: int) -> cp.ndarray:
        psi = cp.zeros(2**n_qubits, dtype=complex)
        psi[0] = 1.0
        return haar_random_unitary_gpu(2**n_qubits) @ psi

    return


@app.cell
def _(cp, generate_random_generators_symplectic, pauli_string_to_matrix):
    def rand_stab(n_qubits: int):
        generators = generate_random_generators_symplectic(n_qubits)
        dim = 2**n_qubits

        g_matrices = [pauli_string_to_matrix(g) for g in generators]

        for i in range(dim):
            v = cp.zeros(dim, dtype=complex)
            v[i] = 1.0
            is_orthogonal = False

            for G in g_matrices:
                v = (v + G @ v) / 2.0

                if cp.linalg.norm(v) < 1e-10:
                    is_orthogonal = True
                    break

            if not is_orthogonal:
                return v / cp.linalg.norm(v)

        raise ValueError("Generators are invalid (no common +1 eigenstate found).")

    return


@app.cell
def _(cp):
    def haar_random_unitary_gpu(dim: int) -> cp.ndarray:
        z = cp.random.randn(dim, dim) + 1j * cp.random.randn(dim, dim)
        q, r = cp.linalg.qr(z)
        d = cp.diagonal(r)
        ph = d / cp.abs(d)
        return q * ph

    return (haar_random_unitary_gpu,)


@app.cell
def _(np):
    def par_trace(psi, dim, n, n_parties):
        n_rem = int(n - n_parties)
        rho = np.outer(psi, psi.conj().T).reshape(
            (dim**n_rem, dim**n_parties, dim**n_rem, dim**n_parties)
        )
        return np.trace(rho, axis1=1, axis2=3)

    return (par_trace,)


@app.cell
def _(cp):
    def inv_perm(permn):
        s = cp.zeros(len(permn), dtype=int)
        s[cp.array(permn)] = list(cp.arange(len(permn)))
        return s.tolist()

    return (inv_perm,)


@app.cell
def _(cp, jl, np):
    def compute_sre(psi_np, alpha=2):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dim = len(psi_np)
                n_qubits = int(np.log2(dim))

                jl.psi_python = cp.asnumpy(psi_np)
                jl.alpha = alpha
                jl.n_qubits = n_qubits
                jl.dim = dim

                jl.seval(""" 
                    psi_jl = Vector{ComplexF64}(psi_python)
                    psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, n_qubits, dim)
                    sre_result, lost_norm = SRE(psi_sv, alpha, backend= :CUDA)
                """)

                sre_result = jl.sre_result
                lost_norm = jl.lost_norm

                return sre_result, lost_norm

            except Exception as e:
                print(f"Execution Error: {e}")
                return None, None

    return (compute_sre,)


@app.cell
def _():
    # OPTIMIZATION: Cached this expensive computation so it doesn't re-run
    # unless inputs change.
    return


@app.cell
def _(
    compute_sre,
    go,
    make_subplots,
    mo,
    np,
    pc,
    pickle,
    rand_Almost_Stab_state,
):
    def plot_sre_vs_gap(n_qubits: int, num_samples: int) -> go.Figure:
        gaps = list(range(0, n_qubits + 1))
        _sre_all = {gap: [] for gap in gaps}
        _sre_means = []
        _sre_stds = []

        for gap in gaps:
            print(f"Processing almost_gap = {gap} / {n_qubits} ...")
            for i in range(num_samples):
                try:
                    _psi = rand_Almost_Stab_state(n_qubits, gap)
                    _sre_val, _lost_norm = compute_sre(_psi, alpha=2)
                    if _sre_val is not None:
                        _sre_all[gap].append(_sre_val)
                except Exception as e:
                    print(f"  Error at sample {i}: {e}")

            _gap_vals = _sre_all[gap]
            _mean = float(np.mean(_gap_vals))
            _std = float(np.std(_gap_vals))
            _sre_means.append(_mean)
            _sre_stds.append(_std)
            print(f"  Done: mean SRE = {_mean:.4f}  ±  {_std:.4f}")

        _fig = go.Figure()
        _fig.add_trace(
            go.Scatter(
                x=gaps,
                y=_sre_means,
                error_y=dict(
                    type="data",
                    array=_sre_stds,
                    visible=True,
                    thickness=1.5,
                    width=3,
                ),
                mode="markers+lines",
                marker=dict(
                    size=8, color="royalblue", line=dict(color="navy", width=1)
                ),
                line=dict(color="royalblue", width=2),
                name="Mean SRE",
            )
        )

        _fig.update_layout(
            title=dict(
                text=f"Stabilizer Rényi Entropy (α=2) vs Almost-Gap<br><sup>{n_qubits}-qubit system, {num_samples} samples per gap</sup>",
                x=0.5,
            ),
            xaxis_title="Almost Gap (free qubits not projected onto stabilizer)",
            yaxis_title="SRE (α=2)",
            template="plotly_white",
            hovermode="x unified",
            width=700,
            height=500,
            xaxis=dict(dtick=1),
        )

        return _fig


    def hex_to_rgba(hex_color, alpha=0.2):
        hex_color = hex_color.lstrip("#")
        return f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {alpha})"


    def plot_trajectories_marimo(pkl_files, labels, step_mes, central_tendency):
        metrics = ["average_purity", "max_purity", "sre"]
        metric_titles = ["Average Purity", "Max Purity", "SRE"]

        colors = pc.qualitative.Plotly
        fig = make_subplots(rows=1, cols=3, subplot_titles=metric_titles)

        for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
            with open(file, "rb") as f:
                data = pickle.load(f)

            base_color = colors[file_idx % len(colors)]
            fill_color = hex_to_rgba(base_color, alpha=0.2)

            for i, metric in enumerate(metrics):
                col = i + 1
                arr = np.array(data[metric])
                steps = np.arange(arr.shape[1]) * step_mes

                if central_tendency == "Average":
                    center_line = np.mean(arr, axis=0)
                    std = np.std(arr, axis=0)
                    lower_bound = center_line - std
                    upper_bound = center_line + std
                    legend_suffix = "(Avg \u00b11 Std)"
                else:
                    center_line = np.median(arr, axis=0)
                    lower_bound = np.percentile(arr, 25, axis=0)
                    upper_bound = np.percentile(arr, 75, axis=0)
                    legend_suffix = "(Median & IQR)"

                show_leg = i == 0

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=lower_bound,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        legendgroup=label,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=upper_bound,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=fill_color,
                        showlegend=False,
                        legendgroup=label,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=center_line,
                        mode="lines",
                        line=dict(color=base_color, width=2),
                        name=f"{label} {legend_suffix}",
                        legendgroup=label,
                        showlegend=show_leg,
                    ),
                    row=1,
                    col=col,
                )

        fig.update_layout(
            title=f"Entanglement Optimization Trajectories ({central_tendency})",
            height=500,
            width=1200,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
            ),
        )

        fig.update_xaxes(title_text="Optimization Steps")

        return fig


    metric_selector = mo.ui.radio(
        options=["Average", "Median"],
        value="Average",
        label="**Select Central Tendency:** ",
    )
    return metric_selector, plot_sre_vs_gap, plot_trajectories_marimo


@app.cell
def _(
    combinations,
    compute_sre,
    cp,
    inv_perm,
    mean,
    par_trace,
    pickle,
    rand_Almost_Stab_state,
):
    def run_entanglement_optimization(
        n_qubits: int,
        num_starts: int,
        num_loops: int,
        gap: int,
        step_mes: int = 20,
        dim: int = 2,
        filename: str = "trajectory_data.pkl",
    ):
        n = n_qubits
        k = n - n // 2

        trajectories = {
            "average_purity": [],
            "max_purity": [],
            "sre": [],
            "final_states": [],
        }

        for st in range(num_starts):
            psi = rand_Almost_Stab_state(n, gap)
            traj_avg_p = []
            traj_max_p = []
            traj_sre = []

            for ent_step in range(num_loops):
                for _i in combinations(range(n), k):
                    per = list(set(range(n)) - set(_i)) + list(_i)
                    psi = cp.moveaxis(
                        psi.reshape([dim] * n), list(range(n)), per
                    ).flatten()
                    x = par_trace(psi, dim, n, k)

                    x = (x + x.conj().T) / 2.0
                    evals, evecs = cp.linalg.eigh(x)
                    threshold = 1e-12
                    evals_inv_sqrt = cp.where(
                        evals > threshold, evals ** (-0.5), 0.0 + 0j
                    )
                    x_inv_sqrt = evecs @ cp.diag(evals_inv_sqrt) @ evecs.conj().T
                    rho = cp.kron(x_inv_sqrt, cp.eye(dim**k, dtype=complex))

                    psi = rho @ psi
                    psi = cp.moveaxis(
                        psi.reshape([dim] * n), list(range(n)), list(inv_perm(per))
                    ).flatten()
                    psi = psi / cp.sqrt(cp.sum(psi * cp.conj(psi)))

                    # OPTIMIZATION: Explicitly free GPU memory to prevent memory leaks
                    # during massive nested loop computations.
                    del x, evals, evecs, evals_inv_sqrt, x_inv_sqrt, rho

                if ent_step % step_mes == 0:
                    print(f"Random state trajectory: {st + 1}/{num_starts}")
                    s = []
                    for j in combinations(range(n), k):
                        per = list(set(range(n)) - set(j)) + list(j)
                        psi_moved = cp.moveaxis(
                            psi.reshape([dim] * n), list(range(n)), per
                        ).flatten()
                        x_meas = par_trace(psi_moved, dim, n, k)
                        s.append(float(cp.sum(x_meas * cp.conj(x_meas)).real))
                        del x_meas  # Free measurement memory

                    traj_avg_p.append(mean(s))
                    traj_max_p.append(max(s))
                    current_sre = float(compute_sre(psi)[0])
                    traj_sre.append(current_sre)

            trajectories["average_purity"].append(traj_avg_p)
            trajectories["max_purity"].append(traj_max_p)
            trajectories["sre"].append(traj_sre)
            trajectories["final_states"].append(cp.asnumpy(psi))

        with open(filename, "wb") as f:
            pickle.dump(trajectories, f)

        print(f"Saved {num_starts} trajectories to {filename}")
        return trajectories

    return (run_entanglement_optimization,)


@app.cell
def _(run_entanglement_optimization):
    def gen_data(n_qubits: int, num_seeds: int, num_steps: int, f_name: str):
        gaps = list(range(1, 2))
        for _gap in gaps:
            print(
                f"computing for {n_qubits}-qubits with initial random {n_qubits - _gap}-qubit stabilized state."
            )
            _ffname = f_name + str(_gap) + ".pkl"
            run_entanglement_optimization(
                n_qubits,
                num_seeds,
                num_steps,
                _gap,
                step_mes=100,
                filename=_ffname,
            )

    return (gen_data,)


@app.cell
def _(run_sim_btn):
    run_sim_btn
    return


@app.cell
def _(gen_data, mo, run_sim_btn):
    # OPTIMIZATION: Use mo.stop to halt execution until the UI button is pressed.
    mo.stop(
        not run_sim_btn.value,
        mo.md(
            "💡 *Click **Run Simulation** in the top cell to execute the heavy optimization step.*"
        ),
    )
    gen_data(4,100,5000,f_name="data/4_qbt_100_sds_ptmzng_fr_5000_stps")
    return


@app.cell
def _(metric_selector):
    metric_selector
    return


@app.cell
def _(metric_selector, mo, plot_trajectories_marimo):
    # OPTIMIZATION: Wrap the expensive render in mo.lazy()
    def render_trajectory_plots():
        return plot_trajectories_marimo(
            pkl_files=[
                "data/4_qbt_100_sds_ptmzng_fr_5000_stps0.pkl",
                "data/4_qbt_100_sds_ptmzng_fr_5000_stps1.pkl",
                "data/4_qbt_100_sds_ptmzng_fr_5000_stps2.pkl",
                "data/4_qbt_100_sds_ptmzng_fr_5000_stps3.pkl",
                "data/4_qbt_100_sds_ptmzng_fr_5000_stps4.pkl",
            ],
            labels=["gap0", "gap1", "gap2", "gap3", "gap4"],
            step_mes=100,
            central_tendency=metric_selector.value,
        )


    mo.lazy(render_trajectory_plots)
    return


@app.cell
def _(mo, plot_sre_vs_gap):
    # OPTIMIZATION: Lazy loading the cached plot
    def render_sre_plot():
        return plot_sre_vs_gap(4, 1000)


    mo.lazy(render_sre_plot)
    return


@app.cell
def _(np, pickle):
    import os

    pkl_files = [
        "data/4_qbt_100_sds_ptmzng_fr_5000_stps0.pkl",
        "data/4_qbt_100_sds_ptmzng_fr_5000_stps1.pkl",
        "data/4_qbt_100_sds_ptmzng_fr_5000_stps2.pkl",
        "data/4_qbt_100_sds_ptmzng_fr_5000_stps3.pkl",
        "data/4_qbt_100_sds_ptmzng_fr_5000_stps4.pkl",
    ]
    labels = ["gap0", "gap1", "gap2", "gap3", "gap4"]
    metrics = ["average_purity", "max_purity", "sre"]

    debug_results = []

    for file, label in zip(pkl_files, labels):
        if not os.path.exists(file):
            debug_results.append(f"{label}: File not found")
            continue

        try:
            with open(file, "rb") as f:
                data = pickle.load(f)

            file_info = {"label": label, "metrics": {}}
            for metric in metrics:
                if metric in data:
                    arr = np.array(data[metric])
                    has_nan = np.isnan(arr).any()
                    shape = arr.shape
                    file_info["metrics"][metric] = (
                        f"shape={shape}, has_nan={has_nan}"
                    )
                else:
                    file_info["metrics"][metric] = "Missing"
            debug_results.append(file_info)
        except Exception as e:
            debug_results.append(f"{label}: Error loading file: {e}")

    for res in debug_results:
        if isinstance(res, str):
            print(res)
        else:
            print(f"{res['label']}:")
            for m, info in res["metrics"].items():
                print(f"  {m}: {info}")
    return


@app.cell
def _(
    combinations,
    compute_sre,
    cp,
    mean,
    mo,
    par_trace,
    rand_Almost_Stab_state,
    run_diag_btn,
):
    import traceback

    # OPTIMIZATION: Halt execution of this expensive diagnostic unless manually triggered
    mo.stop(
        not run_diag_btn.value,
        mo.md(
            "💡 *Click **Run Diagnostics** in the top cell to execute the diagnostic suite.*"
        ),
    )

    diagnostic_results = []
    _n_qubits_test = 4

    for _gap in range(0, _n_qubits_test + 1):
        _gap_info = {"gap": _gap, "stages": {}}
        try:
            _psi_test = rand_Almost_Stab_state(_n_qubits_test, _gap)
            _has_nan_psi = cp.isnan(_psi_test).any()
            _norm_psi = float(cp.linalg.norm(_psi_test))
            _gap_info["stages"]["generation"] = {
                "status": "Success",
                "norm": _norm_psi,
                "has_nan": bool(_has_nan_psi),
            }

            if not _has_nan_psi:
                try:
                    _k_test = _n_qubits_test - _n_qubits_test // 2
                    _dim_test = 2
                    _purities = []
                    for _j in combinations(range(_n_qubits_test), _k_test):
                        _per = list(set(range(_n_qubits_test)) - set(_j)) + list(
                            _j
                        )
                        _psi_moved = cp.moveaxis(
                            _psi_test.reshape([_dim_test] * _n_qubits_test),
                            list(range(_n_qubits_test)),
                            _per,
                        ).flatten()
                        _x_temp = par_trace(
                            _psi_moved, _dim_test, _n_qubits_test, _k_test
                        )
                        _purities.append(
                            float(cp.sum(_x_temp * cp.conj(_x_temp)).real)
                        )
                    _avg_purity = mean(_purities)
                    _max_purity = max(_purities)
                    _gap_info["stages"]["purity"] = {
                        "status": "Success",
                        "avg_purity": _avg_purity,
                        "max_purity": _max_purity,
                    }
                except Exception as _pur_err:
                    _gap_info["stages"]["purity"] = {
                        "status": "Failed",
                        "error": str(_pur_err),
                        "traceback": traceback.format_exc(),
                    }
            else:
                _gap_info["stages"]["purity"] = {"status": "Skipped (NaN state)"}

            if not _has_nan_psi:
                try:
                    _sre_val, _lost_norm = compute_sre(_psi_test, alpha=2)
                    _gap_info["stages"]["sre"] = {
                        "status": "Success"
                        if _sre_val is not None
                        else "Returned None",
                        "sre_val": _sre_val,
                        "lost_norm": _lost_norm,
                    }
                except Exception as _sre_err:
                    _gap_info["stages"]["sre"] = {
                        "status": "Failed",
                        "error": str(_sre_err),
                        "traceback": traceback.format_exc(),
                    }
            else:
                _gap_info["stages"]["sre"] = {"status": "Skipped (NaN state)"}

            _k_test = _n_qubits_test - _n_qubits_test // 2
            _dim_test = 2
            try:
                _per_test = list(range(_n_qubits_test))
                _x_test = par_trace(_psi_test, _dim_test, _n_qubits_test, _k_test)
                _x_test = (_x_test + _x_test.conj().T) / 2.0
                _evals_test, _evecs_test = cp.linalg.eigh(_x_test)

                _gap_info["stages"]["optimization_step"] = {
                    "status": "Success",
                    "min_eval": float(cp.min(_evals_test)),
                    "max_eval": float(cp.max(_evals_test)),
                    "has_nan_evals": bool(cp.isnan(_evals_test).any()),
                }
            except Exception as _opt_err:
                _gap_info["stages"]["optimization_step"] = {
                    "status": "Failed",
                    "error": str(_opt_err),
                }

        except Exception as _gen_err:
            _gap_info["stages"]["generation"] = {
                "status": "Failed",
                "error": str(_gen_err),
                "traceback": traceback.format_exc(),
            }

        diagnostic_results.append(_gap_info)


    # OPTIMIZATION: Render UI output lazily
    def render_diagnostics():
        return mo.vstack(
            [
                mo.md("### Diagnostic Results for Gap Issues (4 Qubits)"),
                mo.ui.table(
                    [
                        {
                            "Gap": res["gap"],
                            "Gen Status": res["stages"]["generation"]["status"],
                            "Gen NaN": res["stages"]["generation"].get(
                                "has_nan", "N/A"
                            ),
                            "Avg Purity": res["stages"]
                            .get("purity", {})
                            .get("avg_purity", "N/A"),
                            "Max Purity": res["stages"]
                            .get("purity", {})
                            .get("max_purity", "N/A"),
                            "SRE Status": res["stages"]
                            .get("sre", {})
                            .get("status", "N/A"),
                            "SRE Val": res["stages"]
                            .get("sre", {})
                            .get("sre_val", "N/A"),
                            "SRE Error": res["stages"]
                            .get("sre", {})
                            .get("error", "None"),
                        }
                        for res in diagnostic_results
                    ]
                ),
                mo.accordion(
                    {"Detailed Diagnostic JSON Tree": mo.tree(diagnostic_results)}
                ),
            ]
        )


    mo.lazy(render_diagnostics)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
