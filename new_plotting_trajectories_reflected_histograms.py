# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "cupy-cuda12x",
#     "plotly",
#     "scipy",
#     "fsspec",
#     "requests",
#     "juliacall==0.9.35",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    try:
        from juliacall import Main as jl
    except ImportError:
        jl = None
    return (jl,)


@app.cell
def _(jl, mo):
    if jl is not None:
        jl.seval("using HadaMAG")
        jl.seval("using CUDA")
        jl.seval("""
        function jl_compute_sre_exact(psi_np, alpha, n_qubits, dim)
            psi_jl = Vector{ComplexF64}(psi_np)
            psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, Int(n_qubits), Int(dim))
            res = SRE(psi_sv, Int(round(alpha)); progress=false)
            return (Float64(res[1]), Float64(res[2]))
        end
        """)
        status = mo.md("⚡ **Julia SRE Direct Bridge:** Initialized successfully (`HadaMAG` + `CUDA` backend)")
    else:
        status = mo.md("⚠️ **Julia SRE Direct Bridge:** Julia/JuliaCall not available (falling back to placeholders)")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cupy as cp
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc
    import pickle
    from itertools import combinations
    import os
    from fsspec.implementations.github import GithubFileSystem
    import warnings
    import matplotlib.pyplot as plt

    # Suppress all warnings globally to avoid logging overhead in loops
    warnings.filterwarnings("ignore")

    _CACHE_FILE = "_plot_cache.pkl"

    class _DiskCache(dict):
        def __init__(self, path: str):
            super().__init__()
            self._path = path
            self._load()

        def _load(self):
            if os.path.exists(self._path):
                try:
                    with open(self._path, "rb") as _f:
                        self.update(pickle.load(_f))
                    print(f"[cache] Loaded {len(self)} entries from {self._path}")
                except Exception as _e:
                    print(f"[cache] Could not load disk cache: {_e}. Starting fresh.")

        def _save(self):
            try:
                with open(self._path, "wb") as _f:
                    pickle.dump(dict(self), _f)
            except Exception as _e:
                print(f"[cache] Could not save to disk: {_e}")

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            self._save()

        def clear(self):
            super().clear()
            try:
                if os.path.exists(self._path):
                    os.remove(self._path)
                    print(f"[cache] Removed disk cache file {self._path}")
            except Exception as _e:
                print(f"[cache] Could not remove disk cache: {_e}")

    # One global instance — loaded from disk once per kernel session
    plot_cache = globals().get("plot_cache", _DiskCache(_CACHE_FILE))
    return (
        GithubFileSystem,
        combinations,
        cp,
        go,
        make_subplots,
        mo,
        np,
        os,
        pc,
        pickle,
        plot_cache,
        plt,
    )


@app.cell
def _(cp):
    def par_trace(psi, dim, n, n_parties):
        n_rem = n - n_parties
        psi_mat = psi.reshape(dim**n_rem, dim**n_parties)
        return psi_mat @ psi_mat.conj().T

    def is_appt(x) -> bool:
        xp = cp.get_array_module(x)
        _purity = xp.sum(x.real**2 + x.imag**2)
        _D = x.shape[0]
        if _purity <= 1 / (_D - 1):
            return True
        _ex = xp.linalg.eigvalsh(x)
        _ex = xp.clip(_ex, 0.0, None)
        if (_ex[-1] - _ex[1]) ** 2 <= 4 * _ex[0] * _ex[3]:
            return True
        return False

    return is_appt, par_trace


@app.cell
def _(combinations, cp, is_appt, np, par_trace):
    def is_TE(psi, dim: int = 2) -> bool:
        xp = cp.get_array_module(psi)
        n = int(np.log2(len(psi)))
        k = n - n // 2
        for _i in combinations(range(n), k):
            per = [x for x in range(n) if x not in _i] + list(_i)
            psi_moved = xp.transpose(psi.reshape([dim] * n), per).flatten()
            _x = par_trace(psi_moved, dim, n, k)
            _x = (_x + _x.conj().T) / 2.0
            if not is_appt(_x):
                return False
        return True

    return (is_TE,)


@app.function
def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip("#")
    return f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {alpha})"


@app.cell
def _(
    go,
    is_TE,
    jl,
    make_subplots,
    np,
    opt_len,
    os,
    pc,
    pickle,
    plot_cache,
    re,
):
    def compute_sre_exact(psi_np, alpha=2):
        """Compute exact SRE using HadaMAG.jl via JuliaCall."""
        if jl is None:
            return 1e-15, 0.0
        try:
            arr_np = np.asarray(psi_np)
            norm_val = np.linalg.norm(arr_np)
            if norm_val > 1e-12:
                arr_np = arr_np / norm_val
            dim = len(arr_np)
            n_qubits = int(np.log2(dim))

            # Call pre-defined Julia function directly — no seval parser overhead per iteration
            res = jl.jl_compute_sre_exact(arr_np, alpha, n_qubits, dim)
            val = float(res[0])
            return val if val != 0.0 else 1e-15, float(res[1])
        except Exception as e:
            print(f"SRE Exact Calculation Error: {e}")
            return 1e-15, 0.0

    def parse_gap_sre(file, label=""):
        target_str = f"{label} {file}"
        _match = re.search(r'(?:gap\s*|k\s*=\s*|_)?(\d+)\.pkl$', file or "")
        if not _match:
            _match = re.search(r'(?:gap\s*|k\s*=\s*)(\d+)', target_str, re.IGNORECASE)
        if _match:
            val = float(_match.group(1))
            if val <= 15:
                return val
        return 0.0

    plot_cache.clear()
    print("compute_sre_exact reloaded OK & plot_cache cleared")

    def get_state_mask(final_states, filter_opt):
        te_flags = np.array([is_TE(state) for state in final_states])
        if filter_opt == "TE States" or filter_opt is True:
            return te_flags, f"TE-only (n={te_flags.sum()}) "
        elif filter_opt == "Non-TE States":
            non_te = ~te_flags
            return non_te, f"Non-TE-only (n={non_te.sum()}) "
        else:
            return np.ones(len(final_states), dtype=bool), ""

    def cache_plot(func):
        """Decorator: cache plotting results keyed on params + file modification times."""
        def wrapper(pkl_files, labels, step_mes, use_te_filter, central_tendency,
                    selected_metrics, plot_type="Line Chart", fs=None):
            mtimes = []
            if not fs:
                for f in (pkl_files or []):
                    if f and os.path.exists(f):
                        mtimes.append(os.path.getmtime(f))
                    else:
                        mtimes.append(0.0)
            cache_key = (
                tuple(pkl_files) if pkl_files else (),
                tuple(labels) if labels else (),
                step_mes,
                use_te_filter,
                central_tendency,
                tuple(selected_metrics) if selected_metrics else (),
                plot_type,
                tuple(mtimes),
            )
            if cache_key in plot_cache:
                print("[cache] HIT — returning cached plot.")
                return plot_cache[cache_key]
            print("[cache] MISS — computing plot…")
            fig = func(pkl_files, labels, step_mes, use_te_filter, central_tendency,
                       selected_metrics, plot_type, fs)
            if fig is not None:
                plot_cache[cache_key] = fig
            return fig
    def plot_te_filtered_trajectories(
        pkl_files, labels, step_mes, use_te_filter, central_tendency, selected_metrics, plot_type="Line Chart", fs=None
    ):
        """Plot trajectories with optional TE filtering and bar chart modes."""
        def finalize_arxiv_style(fig):
            if fig is None:
                return fig
            fig.update_xaxes(
                showline=True, linewidth=1, linecolor="black", mirror=True,
                ticks="inside", tickwidth=1, tickcolor="black",
                gridcolor="#e0e0e0", gridwidth=0.5, zeroline=False
            )
            fig.update_yaxes(
                showline=True, linewidth=1, linecolor="black", mirror=True,
                ticks="inside", tickwidth=1, tickcolor="black",
                gridcolor="#e0e0e0", gridwidth=0.5, zeroline=False
            )
            return fig

        if not selected_metrics:
            return None

        metric_map = {
            "average_purity": "Average Purity",
            "max_purity": "Max Purity",
            "sre": "SRE"
        }
        metric_titles = [metric_map[m] for m in selected_metrics]
        colors = pc.qualitative.Plotly

        # ------------------------------------------------------------
        # BAR CHART COMPARING INITIAL AND FINAL METRICS
        # ------------------------------------------------------------
        if plot_type == "Bar Chart":
            fig = make_subplots(rows=1, cols=len(selected_metrics), subplot_titles=metric_titles)

            for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
                if not file or not label:
                    continue
                try:
                    if fs:
                        with fs.open(file, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(file, "rb") as f:
                            data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

                te_mask, prefix = get_state_mask(data["final_states"], use_te_filter)
                if te_mask.sum() == 0:
                    continue

                base_color = colors[file_idx % len(colors)]

                derived_init_sre = parse_gap_sre(file, label)

                data_changed = False
                for i, metric in enumerate(selected_metrics):
                    col = i + 1

                    init_vals = []
                    final_vals = []

                    for j, keep in enumerate(te_mask):
                        if not keep:
                            continue

                        if metric == "sre":
                            final_state = data["final_states"][j]
                            init_sre = data["sre"][j][0]
                            final_sre = data["sre"][j][-1]

                            if init_sre == 0.0 and derived_init_sre != 0.0:
                                init_sre = derived_init_sre
                                data["sre"][j][0] = derived_init_sre
                                data_changed = True
                            if final_sre == 0.0:
                                computed_final, _ = compute_sre_exact(final_state, alpha=2)
                                final_sre = computed_final
                                data["sre"][j][-1] = computed_final
                                data_changed = True
                            init_vals.append(init_sre)
                            final_vals.append(final_sre)
                        else:
                            traj = data[metric][j]
                            init_vals.append(traj[0])
                            final_vals.append(traj[-1])

                    # Average or Median of Initial vs Final
                    if central_tendency == "Average":
                        init_center = np.mean(init_vals)
                        final_center = np.mean(final_vals)
                        init_err = np.std(init_vals)
                        final_err = np.std(final_vals)
                    else:
                        init_center = np.median(init_vals)
                        final_center = np.median(final_vals)
                        init_err = np.nanstd(init_vals)
                        final_err = np.nanstd(final_vals)

                    # Add Initial Bar
                    fig.add_trace(
                        go.Bar(
                            x=[f"{label} Initial"],
                            y=[init_center],
                            error_y=dict(type='data', array=[init_err], visible=True),
                            name=f"{label} (Initial)",
                            legendgroup=label,
                            showlegend=(i == 0),
                            marker_color=base_color,
                            opacity=0.6
                        ),
                        row=1, col=col
                    )

                    # Add Final Bar
                    fig.add_trace(
                        go.Bar(
                            x=[f"{label} Final"],
                            y=[final_center],
                            error_y=dict(type='data', array=[final_err], visible=True),
                            name=f"{label} (Final)",
                            legendgroup=label,
                            showlegend=False,
                            marker_color=base_color,
                            opacity=1.0
                        ),
                        row=1, col=col
                    )

            # Store back to pickle (once per file, after all metrics processed)
            if data_changed and not fs:
                try:
                    with open(file, "wb") as f:
                        pickle.dump(data, f)
                except Exception as e:
                    print(f"Error saving updated SRE back to {file}: {e}")

            fig.update_layout(
                title=f"Comparison: Initial vs Final States ({central_tendency})",
                height=500,
                width=400 * len(selected_metrics),
                template="plotly_white",
                margin=dict(b=120),
                font=dict(family="Computer Modern"),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                ),
            )
            return finalize_arxiv_style(fig)

        # ------------------------------------------------------------
        # STANDARD LINE CHART TRAJECTORIES
        # ------------------------------------------------------------
        if plot_type == "Line Chart":
            fig = make_subplots(rows=1, cols=len(selected_metrics), subplot_titles=metric_titles)

            for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
                if not file or not label:
                    continue
                try:
                    if fs:
                        with fs.open(file, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(file, "rb") as f:
                            data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

                # Filter by state type if requested (TE / Non-TE / All)
                te_mask, prefix = get_state_mask(data["final_states"], use_te_filter)
                if te_mask.sum() == 0:
                    continue

                base_color = colors[file_idx % len(colors)]
                fill_color = hex_to_rgba(base_color, alpha=0.2)

                derived_init_sre = parse_gap_sre(file, label)

                data_changed = False
                for i, metric in enumerate(selected_metrics):
                    col = i + 1

                    filtered_trajs = []

                    for j, keep in enumerate(te_mask):
                        if not keep:
                            continue
                        if metric == "sre":
                            final_state = data["final_states"][j]
                            traj = data["sre"][j]
                            init_sre = traj[0]
                            final_sre = traj[-1]

                            if init_sre == 0.0 and derived_init_sre != 0.0:
                                init_sre = derived_init_sre
                                data["sre"][j][0] = derived_init_sre
                                data_changed = True
                            if final_sre == 0.0:
                                computed_final, _ = compute_sre_exact(final_state, alpha=2)
                                final_sre = computed_final
                                data["sre"][j][-1] = computed_final
                                data_changed = True

                            has_step_by_step = (len(traj) > 2 and any(v != 0.0 for v in traj[1:-1]))
                            if has_step_by_step:
                                t_copy = list(traj)
                                if t_copy[0] == 0.0 and derived_init_sre != 0.0:
                                    t_copy[0] = derived_init_sre
                                if t_copy[-1] == 0.0:
                                    t_copy[-1] = computed_final
                                filtered_trajs.append(t_copy)
                            else:
                                filtered_trajs.append(list(np.linspace(init_sre, final_sre, opt_len)))
                        else:
                            filtered_trajs.append(data[metric][j])

                    # Store back to pickle
                    if data_changed and not fs:
                        try:
                            with open(file, "wb") as f:
                                pickle.dump(data, f)
                        except Exception as e:
                            print(f"Error saving updated SRE back to {file}: {e}")

                    if not filtered_trajs:
                        continue
                    max_len = max(len(t) for t in filtered_trajs)
                    arr = np.full((len(filtered_trajs), max_len), np.nan)
                    for j, traj in enumerate(filtered_trajs):
                        arr[j, :len(traj)] = traj

                    if max_len == 2:
                        steps = np.array([0, 1]) * step_mes
                    else:
                        steps = np.arange(max_len) * step_mes

                    if central_tendency == "Average":
                        center_line = np.nanmean(arr, axis=0)
                        std = np.nanstd(arr, axis=0)
                        lower_bound = center_line - std
                        upper_bound = center_line + std
                        leg_suffix = f"(Avg ±1 Std)"
                    else:
                        center_line = np.nanmedian(arr, axis=0)
                        lower_bound = np.nanpercentile(arr, 25, axis=0)
                        upper_bound = np.nanpercentile(arr, 75, axis=0)
                        leg_suffix = f"(Median & IQR)"

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
                            name=f"{prefix}{label} {leg_suffix}",
                            legendgroup=label,
                            showlegend=show_leg,
                        ),
                        row=1,
                        col=col,
                    )

                # Store back to pickle (once per file, after all metrics processed)
                if data_changed and not fs:
                    try:
                        with open(file, "wb") as f:
                            pickle.dump(data, f)
                    except Exception as e:
                        print(f"Error saving updated SRE back to {file}: {e}")

            fig.update_layout(
                title=f"Entanglement Trajectories ({central_tendency})"
                + (f" — {use_te_filter}" if use_te_filter in ["TE States", "Non-TE States"] or use_te_filter is True else ""),
                height=500,
                width=400 * len(selected_metrics),
                hovermode="x unified",
                template="plotly_white",
                margin=dict(b=120),
                font=dict(family="Computer Modern"),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                ),
            )
            fig.update_xaxes(title_text="Optimization Steps")
            return finalize_arxiv_style(fig)

        # ------------------------------------------------------------
        # TE vs non-TE REFLECTED SRE   & COUNTS   COMPARISON
        # ------------------------------------------------------------
        if plot_type == "TE vs non-TE SRE":
            fig = go.Figure()

            x_ticks = []
            init_centers = []
            init_errs = []
            init_counts = []

            te_centers = []
            te_errs = []
            te_counts = []

            non_te_centers = []
            non_te_errs = []
            non_te_counts = []

            global_init_sre_vals = []
            global_te_sre_vals = []
            global_non_te_sre_vals = []

            for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
                if not file or not label:
                    continue
                try:
                    if fs:
                        with fs.open(file, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(file, "rb") as f:
                            data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

                final_states = data["final_states"]
                te_mask = np.array([is_TE(state) for state in final_states])

                init_sre_vals = []
                te_sre_vals = []
                non_te_sre_vals = []

                derived_init_sre = parse_gap_sre(file, label)

                for j, is_te_state in enumerate(te_mask):
                    init_sre = data["sre"][j][0]
                    sre_val = data["sre"][j][-1]

                    if init_sre == 0.0 and derived_init_sre != 0.0:
                        init_sre = derived_init_sre
                    if sre_val == 0.0:
                        computed, _ = compute_sre_exact(data["final_states"][j], alpha=2)
                        sre_val = computed

                    init_sre_vals.append(init_sre)
                    if is_te_state:
                        te_sre_vals.append(sre_val)
                    else:
                        non_te_sre_vals.append(sre_val)

                global_init_sre_vals.extend(init_sre_vals)
                global_te_sre_vals.extend(te_sre_vals)
                global_non_te_sre_vals.extend(non_te_sre_vals)

                x_ticks.append(label)

                init_centers.append(np.mean(init_sre_vals) if central_tendency == "Average" else np.median(init_sre_vals))
                init_errs.append(np.std(init_sre_vals) if len(init_sre_vals) > 1 else 0.0)
                init_counts.append(len(init_sre_vals))

                if te_sre_vals:
                    te_centers.append(np.mean(te_sre_vals) if central_tendency == "Average" else np.median(te_sre_vals))
                    te_errs.append(np.std(te_sre_vals) if len(te_sre_vals) > 1 else 0.0)
                    te_counts.append(len(te_sre_vals))
                else:
                    te_centers.append(0.0); te_errs.append(0.0); te_counts.append(0)

                if non_te_sre_vals:
                    non_te_centers.append(np.mean(non_te_sre_vals) if central_tendency == "Average" else np.median(non_te_sre_vals))
                    non_te_errs.append(np.std(non_te_sre_vals) if len(non_te_sre_vals) > 1 else 0.0)
                    non_te_counts.append(len(non_te_sre_vals))
                else:
                    non_te_centers.append(0.0); non_te_errs.append(0.0); non_te_counts.append(0)

            x_ticks.append("All Gaps Combined")
            init_centers.append(np.mean(global_init_sre_vals) if central_tendency == "Average" else np.median(global_init_sre_vals))
            init_errs.append(np.std(global_init_sre_vals) if len(global_init_sre_vals) > 1 else 0.0)
            init_counts.append(len(global_init_sre_vals))

            if global_te_sre_vals:
                te_centers.append(np.mean(global_te_sre_vals) if central_tendency == "Average" else np.median(global_te_sre_vals))
                te_errs.append(np.std(global_te_sre_vals) if len(global_te_sre_vals) > 1 else 0.0)
                te_counts.append(len(global_te_sre_vals))
            else:
                te_centers.append(0.0); te_errs.append(0.0); te_counts.append(0)

            if global_non_te_sre_vals:
                non_te_centers.append(np.mean(global_non_te_sre_vals) if central_tendency == "Average" else np.median(global_non_te_sre_vals))
                non_te_errs.append(np.std(global_non_te_sre_vals) if len(global_non_te_sre_vals) > 1 else 0.0)
                non_te_counts.append(len(global_non_te_sre_vals))
            else:
                non_te_centers.append(0.0); non_te_errs.append(0.0); non_te_counts.append(0)

            # Scale proportions to negative y-axis
            max_sre = max(max(init_centers or [5.0]), max(te_centers or [5.0]), max(non_te_centers or [5.0]))
            prop_scale = max_sre * 0.75

            te_props_list = []
            non_te_props_list = []
            for tc, nc in zip(te_counts, non_te_counts):
                tot = tc + nc
                if tot > 0:
                    te_props_list.append(tc / tot)
                    non_te_props_list.append(nc / tot)
                else:
                    te_props_list.append(0.0)
                    non_te_props_list.append(0.0)

            te_neg = [-p * prop_scale for p in te_props_list]
            non_te_neg = [-p * prop_scale for p in non_te_props_list]

            # +y SRE & -y Proportion Traces aligned: Green -> Blue -> Light Blue -> Red -> Light Red
            fig.add_trace(go.Bar(x=x_ticks, y=init_centers, error_y=dict(type='data', array=init_errs, visible=True), name="Initial SRE  ", marker_color="mediumseagreen", opacity=0.9))
            fig.add_trace(go.Bar(x=x_ticks, y=te_centers, error_y=dict(type='data', array=te_errs, visible=True), name="Final TE SRE  ", marker_color="royalblue", opacity=0.9))
            fig.add_trace(go.Bar(x=x_ticks, y=te_neg, name="TE Proportion  ", marker_color="cornflowerblue", opacity=0.85))
            fig.add_trace(go.Bar(x=x_ticks, y=non_te_centers, error_y=dict(type='data', array=non_te_errs, visible=True), name="Final non-TE SRE  ", marker_color="crimson", opacity=0.9))
            fig.add_trace(go.Bar(x=x_ticks, y=non_te_neg, name="non-TE Proportion  ", marker_color="lightcoral", opacity=0.85))

            prop_ticks = [0.25, 0.50, 0.75, 1.00]
            pos_yticks = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            neg_yticks = [-p * prop_scale for p in prop_ticks]

            all_yticks = neg_yticks[::-1] + pos_yticks
            all_yticklabels = [f"{int(p*100)}%" for p in prop_ticks[::-1]] + [f"{int(s)}" for s in pos_yticks]

            fig.update_layout(
                title=f"SRE   & Reflected State Proportions  : Initial, TE, and non-TE",
                xaxis_title="Stabilizer gap",
                yaxis_title="Proportion    ←  0  →  SRE  ",
                barmode="group",
                hovermode="x unified",
                template="plotly_white",
                margin=dict(b=120),
                font=dict(family="Computer Modern"),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                yaxis=dict(tickmode='array', tickvals=all_yticks, ticktext=all_yticklabels)
            )
            return finalize_arxiv_style(fig)

        # ------------------------------------------------------------
        # TE vs non-TE PROPORTIONS (2 BARS PER GAP)
        # ------------------------------------------------------------
        if plot_type in ["TE vs non-TE Proportions", "Reflected TE vs non-TE Proportions"]:
            fig = go.Figure()

            x_ticks = []
            te_props = []
            non_te_props = []

            global_te_count = 0
            global_non_te_count = 0

            for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
                if not file or not label:
                    continue
                try:
                    if fs:
                        with fs.open(file, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(file, "rb") as f:
                            data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

                final_states = data["final_states"]
                te_mask = np.array([is_TE(state) for state in final_states])
                n_te = int(np.sum(te_mask))
                n_non_te = int(len(te_mask) - n_te)
                n_total = len(te_mask)

                if n_total == 0:
                    continue

                val_k = label.split("=")[-1].strip() if "=" in label else label
                x_ticks.append(f"k = {val_k}" if not label.startswith("k") else label)

                te_props.append(n_te / n_total)
                non_te_props.append(n_non_te / n_total)

                global_te_count += n_te
                global_non_te_count += n_non_te

            global_total = global_te_count + global_non_te_count
            if global_total > 0:
                x_ticks.append("All Gaps Combined")
                te_props.append(global_te_count / global_total)
                non_te_props.append(global_non_te_count / global_total)

            # 2 Bars per Gap
            fig.add_trace(
                go.Bar(
                    x=x_ticks,
                    y=te_props,
                    name="TE States",
                    marker_color="royalblue",
                    opacity=0.9
                )
            )
            fig.add_trace(
                go.Bar(
                    x=x_ticks,
                    y=non_te_props,
                    name="non-TE States",
                    marker_color="crimson",
                    opacity=0.9
                )
            )

            fig.update_layout(
                title="Proportions of TE vs non-TE States for Each Gap (k)",
                xaxis_title="Stabilizer gap",
                yaxis_title="Proportion of States",
                barmode="group",
                hovermode="x unified",
                template="plotly_white",
                margin=dict(b=120),
                font=dict(family="Computer Modern"),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                yaxis=dict(tickformat='.0%', range=[0, 1.05])
            )
            return finalize_arxiv_style(fig)

        # ------------------------------------------------------------
        # REFLECTED HISTOGRAM OF SRE VALUES (TE VS NON-TE PROPORTIONS)
        # ------------------------------------------------------------
        if plot_type in ["Histogram SRE", "Reflected Histogram SRE"]:
            fig = go.Figure()

            all_te_sre = []
            all_non_te_sre = []

            for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
                if not file or not label:
                    continue
                try:
                    if fs:
                        with fs.open(file, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(file, "rb") as f:
                            data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

                final_states = data["final_states"]
                te_mask = np.array([is_TE(state) for state in final_states])

                for j, is_te in enumerate(te_mask):
                    s_val = data["sre"][j][-1]
                    if s_val == 0.0:
                        computed_s, _ = compute_sre_exact(final_states[j], alpha=2)
                        s_val = computed_s
                    if is_te:
                        all_te_sre.append(s_val)
                    else:
                        all_non_te_sre.append(s_val)

            total_n = len(all_te_sre) + len(all_non_te_sre)
            if total_n > 0:
                bins = np.linspace(0, max(max(all_te_sre or [5.0]), max(all_non_te_sre or [5.0])), 30)

                te_counts, bin_edges = np.histogram(all_te_sre, bins=bins)
                te_props = te_counts / total_n

                non_te_counts, _ = np.histogram(all_non_te_sre, bins=bins)
                non_te_props = non_te_counts / total_n

                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                bin_width = bin_edges[1] - bin_edges[0]

                # Upward bars for TE
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=te_props,
                        width=bin_width,
                        name="TE States",
                        marker_color="royalblue",
                        opacity=0.85
                    )
                )

                # Downward (negative) bars for non-TE
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=[-p for p in non_te_props],
                        width=bin_width,
                        name="non-TE States",
                        marker_color="crimson",
                        opacity=0.85
                    )
                )

                max_prop = max(max(te_props) if len(te_props) else 0.1, max(non_te_props) if len(non_te_props) else 0.1) * 1.15
                yticks = np.linspace(-max_prop, max_prop, 7)
                yticklabels = [f"{abs(y)*100:.1f}%" for y in yticks]

                fig.update_layout(
                    title="Reflected Histogram of Final SRE: TE vs non-TE Proportions",
                    xaxis_title="Final  S₂)",
                    yaxis_title="Proportion of Total States",
                    barmode="relative",
                    hovermode="x unified",
                    template="plotly_white",
                    margin=dict(b=120),
                    font=dict(family="Computer Modern"),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="center",
                        x=0.5,
                    ),
                    yaxis=dict(
                        tickmode='array',
                        tickvals=yticks,
                        ticktext=yticklabels,
                    )
                )
            return finalize_arxiv_style(fig)


    return compute_sre_exact, parse_gap_sre, plot_te_filtered_trajectories


@app.cell
def _(mo):
    mo.md("""
    # Entanglement Trajectory Visualization
    """)
    return


@app.cell
def _(mo):
    data_source = mo.ui.radio(
        options=["Local", "GitHub"],
        value="Local",
        label="**Data Source:**",
        inline=True,
    )
    refresh_button = mo.ui.button(label="🔄 Refresh List", value=0)
    return data_source, refresh_button


@app.cell
def _(GithubFileSystem, data_source, mo, os, refresh_button):
    refresh_button.value

    fs = None
    available_folders = []

    if data_source.value == "Local":
        candidate_dirs = ["correct_data", "data", "new_data", "more_data", "data_zip_7","zip1"]
        available_folders = [
            d
            for d in candidate_dirs
            if os.path.exists(d)
            and any(
                f.endswith(".pkl")
                for f in os.listdir(d)
                if os.path.isfile(os.path.join(d, f))
            )
        ]
        if not available_folders:
            available_folders = ["."]
    else:
        org = "invariantfields"
        repo = "Interactive_TE_Plots"
        try:
            fs = GithubFileSystem(org=org, repo=repo)
            repo_dirs = ["correct_data", "data", "new_data"]
            for rdir in repo_dirs:
                try:
                    if any(_f.endswith(".pkl") for _f in fs.ls(rdir)):
                        available_folders.append(rdir)
                except Exception:
                    pass
            if not available_folders:
                available_folders = ["data"]
        except Exception as e:
            mo.output.append(mo.md(f"⚠️ Error connecting to GitHub: {e}"))
            available_folders = ["data"]

    folder_selector = mo.ui.dropdown(
        options=available_folders,
        label="📁 **Select Folder / Repository:**",
        value=available_folders[0] if available_folders else None,
    )
    return folder_selector, fs


@app.cell
def _(data_source, folder_selector, fs, mo, os, plot_cache, refresh_button):
    import re

    selected_folder = folder_selector.value or "."

    available_files = []
    if data_source.value == "Local":
        if os.path.exists(selected_folder):
            available_files = [
                os.path.join(selected_folder, _f)
                for _f in os.listdir(selected_folder)
                if _f.endswith(".pkl")
            ]
        available_files.sort()
    else:
        if fs:
            try:
                available_files = sorted(
                    [_f for _f in fs.ls(selected_folder) if _f.endswith(".pkl")]
                )
            except Exception as e:
                mo.output.append(
                    mo.md(f"⚠️ Error listing files in {selected_folder}: {e}")
                )

    groups = {}
    for _f in available_files:
        _basename = os.path.basename(_f)
        _match = re.search(r"^(.*)(?:stps|steps|stps_)(\d+)\.pkl$", _basename)
        if _match:
            _prefix = _match.group(1).rstrip("_")
            _gap = int(_match.group(2))
        else:
            _prefix = _basename.replace(".pkl", "")
            _gap = 0

        if _prefix not in groups:
            groups[_prefix] = []
        groups[_prefix].append((_f, _gap))

    sorted_group_names = sorted(groups.keys())

    grouped_files = {}
    for _g in groups:
        sorted_tuples = sorted(groups[_g], key=lambda x: x[1])
        grouped_files[_g] = [item[0] for item in sorted_tuples]

    group_selector = mo.ui.dropdown(
        options=sorted_group_names,
        label="📊 **Select Available Run:**",
        value=sorted_group_names[0] if sorted_group_names else None,
    )

    te_filter_dropdown = mo.ui.dropdown(
        options=["All States", "TE States", "Non-TE States"],
        value="TE States",
        label="**State Filter:**",
    )

    metric_selector = mo.ui.radio(
        options=["Average", "Median"],
        value="Average",
        label="**Central Tendency:** ",
    )

    metric_options = ["average_purity", "max_purity", "sre"]
    metrics_to_plot = mo.ui.multiselect(
        options=metric_options,
        value=metric_options,
        label="**Metrics to Plot:**",
    )

    plot_type_dropdown = mo.ui.dropdown(
        options=[
            "Line Chart",
            "Bar Chart",
            "TE vs non-TE SRE",
            "TE vs non-TE Proportions",
            "Reflected Histogram SRE",
        ],
        value="TE vs non-TE SRE",
        label="**Plot Type:**",
    )

    step_mes_input = mo.ui.number(
        start=1, stop=1000, step=1, value=1, label="Steps per measurement"
    )

    plot_button = mo.ui.run_button(label="🚀 Generate Plot")

    def clear_cache_callback(_):
        plot_cache.clear()
        mo.status.toast("🧹 Plot cache cleared successfully!")

    clear_cache_button = mo.ui.button(
        label="🧹 Clear Plot Cache", on_click=clear_cache_callback
    )

    ui_layout = mo.vstack([
        mo.md("### 1. Data Source Selection"),
        mo.hstack([data_source, refresh_button], align="center", gap=2),
        mo.md("### 2. Folder / Repository & Available Runs"),
        mo.hstack([folder_selector, group_selector], align="center", gap=2),
        mo.md("### 3. Plot Settings"),
        mo.hstack(
            [
                plot_type_dropdown,
                metrics_to_plot,
                te_filter_dropdown,
                metric_selector,
                step_mes_input,
            ],
            justify="start",
            gap=2,
        ),
        mo.md("### 4. Execution"),
        mo.hstack([plot_button, clear_cache_button], gap=2),
    ])

    ui_layout
    return (
        group_selector,
        grouped_files,
        metric_selector,
        metrics_to_plot,
        plot_type_dropdown,
        re,
        step_mes_input,
        te_filter_dropdown,
    )


@app.cell
def _(
    fs,
    group_selector,
    grouped_files,
    metric_selector,
    metrics_to_plot,
    mo,
    plot_te_filtered_trajectories,
    plot_type_dropdown,
    re,
    step_mes_input,
    te_filter_dropdown,
):
    if not group_selector.value or group_selector.value not in grouped_files or not grouped_files.get(group_selector.value):
        mo.stop(True, mo.md("⚠️ **Please select an available dataset folder or run from Step 2.**"))

    selected_group_files = grouped_files[group_selector.value]

    # Deriving labels representing the gaps from the filenames
    labels = []
    for _f in selected_group_files:
        _match = re.search(r'(?:stps|steps|stps_)(\d+)\.pkl$', _f)
        if _match:
            labels.append(f"k = {_match.group(1)}")
        else:
            labels.append(_f.split("/")[-1].replace(".pkl", ""))

    plot = plot_te_filtered_trajectories(
        pkl_files=selected_group_files,
        labels=labels,
        step_mes=step_mes_input.value,
        use_te_filter=te_filter_dropdown.value,
        central_tendency=metric_selector.value,
        selected_metrics=metrics_to_plot.value,
        plot_type=plot_type_dropdown.value,
        fs=fs,
    )
    plot
    return


@app.cell
def _(os, pickle):
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
            return

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

    return (unpack_pkl_file,)


@app.cell
def _(unpack_pkl_file):
    unpack_pkl_file("packed_7_qbt_2000_sds_1500_stps.pkl", )
    return


@app.cell
def _(compute_sre_exact, is_TE, np, parse_gap_sre, pickle, plt, re):
    def plot_te_filtered_trajectories_matplotlib(
        pkl_files, labels, step_mes, use_te_filter, central_tendency, selected_metrics, plot_type="Line Chart", fs=None
    ):
        def get_state_mask(final_states, filter_opt):
            te_flags = np.array([is_TE(state) for state in final_states])
            if filter_opt == "TE States" or filter_opt is True:
                return te_flags, f"TE-only (n={te_flags.sum()}) "
            elif filter_opt == "Non-TE States":
                non_te = ~te_flags
                return non_te, f"Non-TE-only (n={non_te.sum()}) "
            else:
                return np.ones(len(final_states), dtype=bool), ""

        if not selected_metrics:
            return None

        import matplotlib
        matplotlib.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Computer Modern", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white"
        })

        metric_map = {
            "average_purity": "Average Purity",
            "max_purity": "Max Purity",
            "sre": "SRE"
        }

        loaded_data = []
        for file, label in zip(pkl_files, labels):
            if not file or not label:
                continue
            try:
                if fs:
                    with fs.open(file, "rb") as f:
                        data = pickle.load(f)
                else:
                    with open(file, "rb") as f:
                        data = pickle.load(f)
                loaded_data.append((data, label, file))
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if not loaded_data:
            return None

        # 1. LINE CHART
        if plot_type == "Line Chart":
            n_plots = len(selected_metrics)
            fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5), sharex=True, squeeze=False)
            colors = ["#0052CC", "#FF2A54", "#00875A", "#FFAB00", "#6554C0", "#00B8D9", "#FF5630", "#36B37E"]

            for col_idx, metric in enumerate(selected_metrics):
                ax = axes[0, col_idx]
                ax.set_title(metric_map[metric], fontsize=13)

                for data_idx, (data, label, file) in enumerate(loaded_data):
                    derived_init_sre = parse_gap_sre(file, label)

                    te_mask, prefix = get_state_mask(data["final_states"], use_te_filter)
                    indices = [j for j, keep in enumerate(te_mask) if keep]

                    trajs = []
                    for j in indices:
                        traj = data[metric][j]
                        opt_len = len(traj)
                        if opt_len <= 2:
                            for m in ["average_purity", "max_purity", "total_violation", "sre"]:
                                if m in data and len(data[m]) > j and len(data[m][j]) > 2:
                                    opt_len = len(data[m][j])
                                    break

                        is_sre_metric = (metric == "sre")
                        if is_sre_metric:
                            init_sre = traj[0]
                            final_sre = traj[-1]
                            if init_sre == 0.0 and derived_init_sre != 0.0:
                                init_sre = derived_init_sre
                            if final_sre == 0.0:
                                computed_final, _ = compute_sre_exact(data["final_states"][j], alpha=2)
                                final_sre = computed_final

                            if len(traj) > 2:
                                # Use stored trajectory directly (Gap 0 legitimately has SRE=0, which is correct)
                                t_copy = list(traj)
                                if abs(t_copy[0]) <= 1e-5 and derived_init_sre != 0.0:
                                    t_copy[0] = derived_init_sre
                                if abs(t_copy[-1]) <= 1e-5 and abs(derived_init_sre) > 1e-5:
                                    t_copy[-1] = computed_final
                                trajs.append(t_copy)
                            else:
                                trajs.append(list(np.linspace(init_sre, final_sre, opt_len)))
                        else:
                            trajs.append(traj)

                    if not trajs:
                        continue

                    max_len = max(len(t) for t in trajs)
                    arr = np.full((len(trajs), max_len), np.nan)
                    for j, t in enumerate(trajs):
                        arr[j, :len(t)] = t

                    steps = np.arange(max_len) * step_mes

                    if central_tendency == "Average":
                        center = np.nanmean(arr, axis=0)
                        std = np.nanstd(arr, axis=0)
                        lower_bound = center - std
                        upper_bound = center + std
                    else:
                        center = np.nanmedian(arr, axis=0)
                        lower_bound = np.nanpercentile(arr, 25, axis=0)
                        upper_bound = np.nanpercentile(arr, 75, axis=0)

                    color = colors[data_idx % len(colors)]
                    val_k = label.split("=")[-1].strip()
                    ax.plot(steps, center, label=f"$k={val_k}$", color=color, linewidth=2.0)
                    ax.fill_between(steps, lower_bound, upper_bound, color=color, alpha=0.2, edgecolor="none")

                ax.set_xlabel(r"$\text{Optimization Steps}$", fontsize=11)
                # if col_idx == 0:
                #     ax.set_ylabel(r"$\text{Metric Value}$", fontsize=11)
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
                ax.tick_params(direction="in", top=True, right=True, which="both")
                ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0")
                ax.legend(frameon=True, fontsize=9, loc="best")

            fig.tight_layout()
            return fig

        # 2. BAR CHART — Initial (Magic) vs Final with error bars + All Combined
        elif plot_type == "Bar Chart":
            n_plots = len(selected_metrics)
            fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5.0), squeeze=False, facecolor='white')

            for col_idx, metric in enumerate(selected_metrics):
                ax = axes[0, col_idx]
                ax.set_title(metric_map[metric], fontsize=13)

                x_labels = []
                init_means = []
                init_errs = []
                init_counts = []
                final_means = []
                final_errs = []
                final_counts = []
                global_init_vals = []
                global_final_vals = []

                for data_idx, (data, label, file) in enumerate(loaded_data):
                    te_mask, prefix = get_state_mask(data["final_states"], use_te_filter)

                    init_vals = []
                    final_vals = []

                    derived_init_sre = parse_gap_sre(file, label)

                    for j, keep in enumerate(te_mask):
                        if not keep:
                            continue
                        if metric == "sre":
                            init_sre = data["sre"][j][0]
                            final_sre = data["sre"][j][-1]
                            if final_sre == 0.0:
                                computed_final, _ = compute_sre_exact(data["final_states"][j], alpha=2)
                                final_sre = computed_final
                            init_vals.append(init_sre if init_sre != 0.0 else derived_init_sre)
                            final_vals.append(final_sre)
                        else:
                            init_vals.append(data[metric][j][0])
                            final_vals.append(data[metric][j][-1])

                    if not init_vals:
                        continue

                    val_k = label.split("=")[-1].strip()
                    x_labels.append(f"$k={val_k}$")
                    global_init_vals.extend(init_vals)
                    global_final_vals.extend(final_vals)

                    if central_tendency == "Average":
                        init_means.append(np.mean(init_vals))
                        final_means.append(np.mean(final_vals))
                    else:
                        init_means.append(np.median(init_vals))
                        final_means.append(np.median(final_vals))
                    init_errs.append(np.std(init_vals) if len(init_vals) > 1 else 0.0)
                    final_errs.append(np.std(final_vals) if len(final_vals) > 1 else 0.0)
                    init_counts.append(len(init_vals))
                    final_counts.append(len(final_vals))

                # All Combined summary bar
                if global_init_vals:
                    x_labels.append(r"$\mathrm{All\ Combined}$")
                    if central_tendency == "Average":
                        init_means.append(np.mean(global_init_vals))
                        final_means.append(np.mean(global_final_vals))
                    else:
                        init_means.append(np.median(global_init_vals))
                        final_means.append(np.median(global_final_vals))
                    init_errs.append(np.std(global_init_vals) if len(global_init_vals) > 1 else 0.0)
                    final_errs.append(np.std(global_final_vals) if len(global_final_vals) > 1 else 0.0)
                    init_counts.append(len(global_init_vals))
                    final_counts.append(len(global_final_vals))

                x = np.arange(len(x_labels))
                width = 0.35
                ekw = dict(elinewidth=0.8, capthick=0.8)

                b_init = ax.bar(x - width/2, init_means, width, yerr=init_errs, capsize=3,
                       label=r"$\text{Initial State (Magic)}$", color="mediumseagreen",
                       edgecolor="black", linewidth=0.5, error_kw=ekw)
                b_final = ax.bar(x + width/2, final_means, width, yerr=final_errs, capsize=3,
                       label=r"$\text{Final State}$", color="#ff7f0e",
                       edgecolor="black", linewidth=0.5, error_kw=ekw)

                ax.bar_label(b_init, labels=[f"n={n}" for n in init_counts], fontsize=7, padding=3, color="black")
                ax.bar_label(b_final, labels=[f"n={n}" for n in final_counts], fontsize=7, padding=3, color="black")

                ax.set_facecolor('white')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, fontsize=9)
                ax.set_xlabel(r"$\text{Stabilizer gap } (q)$", fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(r"$\text{Value}$", fontsize=11)
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
                ax.tick_params(direction="in", top=True, right=True, which="both")
                ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")
                ax.legend(frameon=True, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
                ax.margins(y=0.15)

            fig.subplots_adjust(bottom=0.22)
            return fig

        # 3. TE vs non-TE SRE — 3 bars for each gap (Initial SRE, TE SRE, non-TE SRE) + Reflected Counts  
        elif plot_type == "TE vs non-TE SRE":
            fig, ax = plt.subplots(figsize=(9.0, 6.0), facecolor='white')
            ax.set_facecolor('white')

            x_labels = []
            init_centers = []
            init_errs = []
            init_counts = []
            te_centers = []
            te_errs = []
            te_counts = []
            non_te_centers = []
            non_te_errs = []
            non_te_counts = []
            global_init = []
            global_te = []
            global_non_te = []

            for data, label, file in loaded_data:
                te_mask = np.array([is_TE(state) for state in data["final_states"]])

                derived_init_sre = 0.0
                _gap_match = re.search(r'(?:gap|k\s*=\s*)(\d+)', label)
                if not _gap_match and file:
                    _gap_match = re.search(r'_stps_(\d+)', file)
                if _gap_match:
                    derived_init_sre = float(_gap_match.group(1))

                init_vals = []
                te_vals = []
                non_te_vals = []

                for j, is_te_state in enumerate(te_mask):
                    init_sre = data["sre"][j][0]
                    final_sre = data["sre"][j][-1]
                    init_vals.append(init_sre if init_sre != 0.0 else derived_init_sre)
                    if is_te_state:
                        te_vals.append(final_sre)
                    else:
                        non_te_vals.append(final_sre)

                if not te_vals and not non_te_vals:
                    continue

                val_k = label.split("=")[-1].strip()
                x_labels.append(f"$q={val_k}$")
                global_init.extend(init_vals)
                global_te.extend(te_vals)
                global_non_te.extend(non_te_vals)

                def _stat(vals):
                    if not vals:
                        return 0.0, 0.0
                    c = np.mean(vals) if central_tendency == "Average" else np.median(vals)
                    e = np.std(vals) if len(vals) > 1 else 0.0
                    return c, e

                ic, ie = _stat(init_vals)
                tc, te_e = _stat(te_vals)
                nc, ne = _stat(non_te_vals)
                init_centers.append(ic); init_errs.append(ie); init_counts.append(len(init_vals))
                te_centers.append(tc); te_errs.append(te_e); te_counts.append(len(te_vals))
                non_te_centers.append(nc); non_te_errs.append(ne); non_te_counts.append(len(non_te_vals))

            # All Combined
            if global_te or global_non_te:
                x_labels.append(r"$\mathrm{All\ Combined}$")
                ic = np.mean(global_init) if central_tendency == "Average" else np.median(global_init)
                tc = np.mean(global_te) if central_tendency == "Average" else np.median(global_te)
                nc = np.mean(global_non_te) if central_tendency == "Average" else np.median(global_non_te)
                init_centers.append(ic); init_errs.append(np.std(global_init) if global_init else 0.0); init_counts.append(len(global_init))
                te_centers.append(tc); te_errs.append(np.std(global_te) if global_te else 0.0); te_counts.append(len(global_te))
                non_te_centers.append(nc); non_te_errs.append(np.std(global_non_te) if global_non_te else 0.0); non_te_counts.append(len(global_non_te))

            x = np.arange(len(x_labels))
            width = 0.25
            ekw = dict(elinewidth=0.8, capthick=0.8)

            # --- POSITIVE Y-AXIS: Original SRE values with error bars (3 bars for each gap) ---
            b_init = ax.bar(x - width, init_centers, width, yerr=init_errs, capsize=3,
                            label=r"$\text{Initial States SRE  }$", color="mediumseagreen",
                            edgecolor="black", linewidth=0.5, error_kw=ekw)
            b_te_sre = ax.bar(x, te_centers, width, yerr=te_errs, capsize=3,
                              label=r"$\text{Final TE SRE  }$", color="royalblue",
                              edgecolor="black", linewidth=0.5, error_kw=ekw)
            b_non_te_sre = ax.bar(x + width, non_te_centers, width, yerr=non_te_errs, capsize=3,
                                  label=r"$\text{Final non-TE SRE  }$", color="crimson",
                                  edgecolor="black", linewidth=0.5, error_kw=ekw)

            # --- NEGATIVE Y-AXIS: Reflected proportions of TE (Light Blue) and non-TE (Light Red) ---
            max_sre = max(max(init_centers or [5.0]), max(te_centers or [5.0]), max(non_te_centers or [5.0]))
            prop_scale = max_sre * 0.75  # Scale 100% to -0.75 * max_sre

            te_props_list = []
            non_te_props_list = []
            for tc, nc in zip(te_counts, non_te_counts):
                tot = tc + nc
                if tot > 0:
                    te_props_list.append(tc / tot)
                    non_te_props_list.append(nc / tot)
                else:
                    te_props_list.append(0.0)
                    non_te_props_list.append(0.0)

            te_neg_heights = [-p * prop_scale for p in te_props_list]
            non_te_neg_heights = [-p * prop_scale for p in non_te_props_list]

            # 2 bars on -y: TE (Light Blue) and non-TE (Light Red)
            b_te_prop = ax.bar(x, te_neg_heights, width, color="cornflowerblue",
                               edgecolor="black", linewidth=0.5, alpha=0.85, label=r"$\text{TE Proportion  }$")
            b_non_te_prop = ax.bar(x + width, non_te_neg_heights, width, color="lightcoral",
                                   edgecolor="black", linewidth=0.5, alpha=0.85, label=r"$\text{non-TE Proportion  }$")

            # Reference baseline at y=0
            ax.axhline(0, color="black", linewidth=0.9, linestyle="-")

            # Set custom ticks: Positive = SRE values, Negative = Percentages
            pos_yticks = np.linspace(0, np.ceil(max_sre), 5)
            prop_ticks = [0.25, 0.50, 0.75, 1.00]
            neg_yticks = [-p * prop_scale for p in prop_ticks]

            all_yticks = list(neg_yticks[::-1]) + list(pos_yticks)
            all_yticklabels = [f"{int(p*100)}%" for p in prop_ticks[::-1]] + [f"{s:.1f}" if s % 1 != 0 else f"{int(s)}" for s in pos_yticks]

            ax.set_yticks(all_yticks)
            ax.set_yticklabels(all_yticklabels, fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9.5)
            ax.set_xlabel(r"$\text{Stabilizer gap } (q)$", fontsize=11)
            ax.set_ylabel(
        r"$\text{Proportion} \quad \longleftarrow \quad \longrightarrow \quad \text{SRE } (S_2)$",
        va="center",
        ha="right",
        fontsize=10.5,  # Apply the blended transform
        y=0.7,  # Explicitly places the center of the text at data coordinate y = 0
    )
            #ax.set_title(r"$\text{SRE (Magic)   vs Reflected State Proportions   (TE vs non-TE)}$", fontsize=12.5)

            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.tick_params(direction="in", top=True, right=True, which="both")
            ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")

            # Align legend handles so Blue & Light Blue are paired, and Red & Light Red are paired
            legend_handles = [b_te_sre, b_te_prop, b_non_te_sre, b_non_te_prop, b_init]
            ax.legend(handles=legend_handles, frameon=True, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
            ax.margins(x=0.04)

            fig.subplots_adjust(bottom=0.22)
            return fig

        # 3b. TE vs non-TE Proportions — 2 bars for each gap (TE proportion & non-TE proportion)
        elif plot_type in ["TE vs non-TE Proportions", "Reflected TE vs non-TE Proportions"]:
            fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor='white')
            ax.set_facecolor('white')

            x_labels = []
            te_props = []
            non_te_props = []

            global_te_count = 0
            global_non_te_count = 0

            for data, label, file in loaded_data:
                te_mask = np.array([is_TE(state) for state in data["final_states"]])
                n_te = int(np.sum(te_mask))
                n_non_te = int(len(te_mask) - n_te)
                n_total = len(te_mask)

                if n_total == 0:
                    continue

                val_k = label.split("=")[-1].strip()
                x_labels.append(f"$k={val_k}$")

                te_props.append(n_te / n_total)
                non_te_props.append(n_non_te / n_total)

                global_te_count += n_te
                global_non_te_count += n_non_te

            # All Combined
            global_total = global_te_count + global_non_te_count
            if global_total > 0:
                x_labels.append(r"$\mathrm{All\ Combined}$")
                te_props.append(global_te_count / global_total)
                non_te_props.append(global_non_te_count / global_total)

            x = np.arange(len(x_labels))
            width = 0.35

            ax.bar(x - width/2, te_props, width, label=r"$\text{TE States}$", color="royalblue",
                   edgecolor="black", linewidth=0.6, alpha=0.9)
            ax.bar(x + width/2, non_te_props, width, label=r"$\text{non-TE States}$", color="crimson",
                   edgecolor="black", linewidth=0.6, alpha=0.9)

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=10)
            ax.set_xlabel(r"$\text{Stabilizer gap } (q)$", fontsize=11)
            ax.set_ylabel(r"$\text{Proportion of States}$", fontsize=11)
            #ax.set_title(r"$\text{Proportions of TE vs non-TE States for Each Gap } (k)$", fontsize=13)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=9)

            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.tick_params(direction="in", top=True, right=True, which="both")
            ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")
            ax.legend(frameon=True, fontsize=10, loc="upper right")

            fig.tight_layout()
            return fig

        # 4. Reflected Histogram SRE
        elif plot_type in ["Histogram SRE", "Reflected Histogram SRE"]:
            fig, ax = plt.subplots(figsize=(8, 5.0), facecolor='white')
            ax.set_facecolor('white')

            all_te_sre = []
            all_non_te_sre = []

            for data, label, file in loaded_data:
                te_mask = np.array([is_TE(state) for state in data["final_states"]])
                for j, is_te in enumerate(te_mask):
                    s_val = data["sre"][j][-1]
                    if s_val == 0.0:
                        computed_s, _ = compute_sre_exact(data["final_states"][j], alpha=2)
                        s_val = computed_s
                    if is_te:
                        all_te_sre.append(s_val)
                    else:
                        all_non_te_sre.append(s_val)

            total_n = len(all_te_sre) + len(all_non_te_sre)
            if total_n > 0:
                bins = np.linspace(0, max(max(all_te_sre or [5.0]), max(all_non_te_sre or [5.0])), 30)

                # TE histogram pointing up  
                te_counts, bin_edges = np.histogram(all_te_sre, bins=bins)

                # non-TE histogram pointing down  
                non_te_counts, _ = np.histogram(all_non_te_sre, bins=bins)

                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                bin_width = bin_edges[1] - bin_edges[0]

                ax.bar(bin_centers, te_counts, width=bin_width, label=r"$\text{Final TE States  }$",
                       color="royalblue", edgecolor="black", linewidth=0.5, alpha=0.85)
                ax.bar(bin_centers, -non_te_counts, width=bin_width, label=r"$\text{Final non-TE States  }$",
                       color="crimson", edgecolor="black", linewidth=0.5, alpha=0.85)

                ax.axhline(0, color="black", linewidth=0.9)

                max_c = max(max(te_counts) if len(te_counts) else 10, max(non_te_counts) if len(non_te_counts) else 10) * 1.15
                c_step = max(1, int(max_c / 4))
                yticks = np.arange(-int(max_c), int(max_c) + 1, c_step)
                ax.set_yticks(yticks)
                ax.set_yticklabels([f"{abs(y)}" for y in yticks], fontsize=9)

            ax.set_xlabel(r"$\text{Final SRE } (S_2)$", fontsize=11)
            ax.set_ylabel(r"$\text{Count of States (non-TE   } \leftarrow 0 \rightarrow \text{ TE  )}$", fontsize=10.5)
            ax.set_title(r"$\text{Reflected Histogram of Final SRE: TE   vs non-TE  }$", fontsize=12.5)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.tick_params(direction="in", top=True, right=True, which="both")
            ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0")
            ax.legend(frameon=True, fontsize=10, loc="upper right")

            fig.tight_layout()
            return fig


    return (plot_te_filtered_trajectories_matplotlib,)


@app.cell
def _(
    fs,
    group_selector,
    grouped_files,
    metric_selector,
    metrics_to_plot,
    mo,
    plot_te_filtered_trajectories_matplotlib,
    plot_type_dropdown,
    re,
    step_mes_input,
    te_filter_dropdown,
):
    if not group_selector.value or group_selector.value not in grouped_files or not grouped_files.get(group_selector.value):
        mo.stop(True, mo.md("⚠️ **Please select an available dataset folder or run from Step 2.**"))

    _mpl_files = grouped_files[group_selector.value]
    _mpl_labels = []
    for _f in _mpl_files:
        _m = re.search(r'(?:stps|steps|stps_)(\d+)\.pkl$', _f)
        if _m:
            _mpl_labels.append(f"k = {_m.group(1)}")
        else:
            _mpl_labels.append(_f.split("/")[-1].replace(".pkl", ""))

    _mpl_fig = plot_te_filtered_trajectories_matplotlib(
        pkl_files=_mpl_files,
        labels=_mpl_labels,
        step_mes=step_mes_input.value,
        use_te_filter=te_filter_dropdown.value,
        central_tendency=metric_selector.value,
        selected_metrics=metrics_to_plot.value,
        plot_type=plot_type_dropdown.value,
        fs=fs,
    )

    matplotlib_plot = _mpl_fig
    matplotlib_plot
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
