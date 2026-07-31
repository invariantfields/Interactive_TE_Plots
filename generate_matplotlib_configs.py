#!/usr/bin/env python3
import os
import re
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# -------------------------------------------------------------------
# Fast Vectorized Batch Quantum TE Evaluation
# -------------------------------------------------------------------
def is_appt_batch(x_batch):
    N, D, _ = x_batch.shape
    purities = np.sum(x_batch.real**2 + x_batch.imag**2, axis=(1, 2))
    bound = 1.0 / (D - 1.0)
    is_te_mask = (purities <= bound)
    
    rem_idx = np.where(~is_te_mask)[0]
    if len(rem_idx) > 0:
        ex = np.linalg.eigvalsh(x_batch[rem_idx])
        ex = np.clip(ex, 0.0, None)
        cond = (ex[:, -1] - ex[:, 1])**2 <= 4.0 * ex[:, 0] * ex[:, 3]
        is_te_mask[rem_idx] = cond
    return is_te_mask

def is_TE_dataset(states, dim=2):
    states = np.asarray(states)
    N, L = states.shape
    n = int(np.log2(L))
    k = n - n // 2
    n_rem = n - k
    
    te_flags = np.ones(N, dtype=bool)
    
    for _i in combinations(range(n), k):
        active = np.where(te_flags)[0]
        if len(active) == 0:
            break
        sub_states = states[active]
        per = [x for x in range(n) if x not in _i] + list(_i)
        
        reshaped = sub_states.reshape([len(active)] + [dim] * n)
        permuted = np.transpose(reshaped, [0] + [p + 1 for p in per])
        mat = permuted.reshape(len(active), dim**n_rem, dim**k)
        x_batch = mat @ np.swapaxes(mat.conj(), 1, 2)
        x_batch = (x_batch + np.swapaxes(x_batch.conj(), 1, 2)) / 2.0
        
        sub_flags = is_appt_batch(x_batch)
        te_flags[active] = sub_flags
        
    return te_flags

def parse_filename_info(filepath):
    filename = os.path.basename(filepath)
    m = re.search(r'(\d+)_qbt_(\d+)_sds_.*?_(\d+)_stps_(\d+)\.pkl$', filename)
    if m:
        qbt, sds, stps, gap = m.groups()
        return {
            'qbt': int(qbt),
            'sds': int(sds),
            'stps': int(stps),
            'gap': int(gap),
            'label': f"{qbt}q_{sds}sds (q={gap})",
            'short_label': f"q={gap}",
            'tag': f"{qbt}q_{sds}sds"
        }
    
    # Fallback pattern
    m_gap = re.search(r'(\d+)\.pkl$', filename)
    gap_val = int(m_gap.group(1)) if m_gap else 0
    return {
        'qbt': 7,
        'sds': 2000,
        'stps': 2500,
        'gap': gap_val,
        'label': f"7q_2000sds (q={gap_val})",
        'short_label': f"q={gap_val}",
        'tag': "7q_2000sds"
    }

def get_dynamic_step_mes(file_path, traj_len):
    info = parse_filename_info(file_path)
    if info.get('stps') and traj_len > 1:
        return float(info['stps']) / (traj_len - 1)
    return 50.0

def load_dataset_files(data_dir="zip7"):
    files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    loaded = []
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        info = parse_filename_info(f)
        loaded.append((data, info, f))
    return loaded

# -------------------------------------------------------------------
# Config 1 & Config 2: Line Chart for TE / non-TE States (All Metrics)
# -------------------------------------------------------------------
def plot_linechart_config(loaded_data, filter_mode="TE States", output_path="matplotlib_linechart.png"):
    selected_metrics = ["sre", "average_purity", "max_purity"]
    metric_map = {
        "sre": r"$\text{SRE } (S_2)$",
        "average_purity": r"$\text{Average Purity}$",
        "max_purity": r"$\text{Max Purity}$"
    }
    colors = ["#0052CC", "#FF2A54", "#00875A", "#FFAB00", "#6554C0", "#00B8D9"]

    n_plots = len(selected_metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.5), squeeze=False, facecolor="white")

    tag_str = loaded_data[0][1]['tag'] if loaded_data else "7q_2000sds"

    for col_idx, metric in enumerate(selected_metrics):
        ax = axes[0, col_idx]
        ax.set_title(metric_map[metric], fontsize=13, fontweight="bold")

        for data_idx, (data, info, file_path) in enumerate(loaded_data):
            final_states = data["final_states"]
            te_mask = is_TE_dataset(final_states)

            if filter_mode == "TE States":
                indices = [j for j, is_te in enumerate(te_mask) if is_te]
            else:
                indices = [j for j, is_te in enumerate(te_mask) if not is_te]

            if not indices:
                continue

            trajs = []
            for j in indices:
                if metric in data:
                    trajs.append(data[metric][j])

            if not trajs:
                continue

            max_len = max(len(t) for t in trajs)
            step_mes = get_dynamic_step_mes(file_path, max_len)

            arr = np.full((len(trajs), max_len), np.nan)
            for j, t in enumerate(trajs):
                arr[j, :len(t)] = t

            steps = np.arange(max_len) * step_mes

            center = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            lower_bound = center - std
            upper_bound = center + std

            color = colors[data_idx % len(colors)]
            label_text = f"{info['label']} (n={len(indices)})"
            ax.plot(steps, center, label=label_text, color=color, linewidth=2.0)
            ax.fill_between(steps, lower_bound, upper_bound, color=color, alpha=0.2, edgecolor="none")

        ax.set_xlabel(r"$\text{Optimization Steps}$", fontsize=11)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(direction="in", top=True, right=True, which="both")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0")
        ax.legend(frameon=True, fontsize=9, loc="best")

    plt.suptitle(f"{tag_str} Trajectories ({filter_mode} — Average Aggregation)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

# -------------------------------------------------------------------
# Config 3: TE vs non-TE SRE (2 Subplots Upright Stack with hspace=0)
# -------------------------------------------------------------------
def plot_te_vs_non_te_sre_config(loaded_data, output_path="matplotlib_config3_te_vs_non_te_sre.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True, facecolor="white", gridspec_kw={"hspace": 0})
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    tag_str = loaded_data[0][1]['tag'] if loaded_data else "7q_2000sds"

    x_labels = []
    init_centers, init_errs, init_counts = [], [], []
    te_centers, te_errs, te_counts = [], [], []
    non_te_centers, non_te_errs, non_counts = [], [], []
    te_props_list, non_te_props_list = [], []

    global_init, global_te, global_non_te = [], [], []

    for data, info, file_path in loaded_data:
        te_mask = is_TE_dataset(data["final_states"])
        derived_init_sre = info['gap']

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

        x_labels.append(f"$q={info['gap']}$")

        global_init.extend(init_vals)
        global_te.extend(te_vals)
        global_non_te.extend(non_te_vals)

        init_centers.append(np.mean(init_vals) if init_vals else 0.0)
        init_errs.append(np.std(init_vals) if len(init_vals) > 1 else 0.0)
        init_counts.append(len(init_vals))

        te_centers.append(np.mean(te_vals) if te_vals else 0.0)
        te_errs.append(np.std(te_vals) if len(te_vals) > 1 else 0.0)
        te_counts.append(len(te_vals))

        non_te_centers.append(np.mean(non_te_vals) if non_te_vals else 0.0)
        non_te_errs.append(np.std(non_te_vals) if len(non_te_vals) > 1 else 0.0)
        non_counts.append(len(non_te_vals))

        tc = len(te_vals)
        nc = len(non_te_vals)
        tot = tc + nc
        if tot > 0:
            te_props_list.append(tc / tot)
            non_te_props_list.append(nc / tot)
        else:
            te_props_list.append(0.0)
            non_te_props_list.append(0.0)

    # All Combined
    if global_init:
        x_labels.append(r"$\mathrm{All\ Combined}$")
        init_centers.append(np.mean(global_init))
        init_errs.append(np.std(global_init))
        init_counts.append(len(global_init))

        te_centers.append(np.mean(global_te) if global_te else 0.0)
        te_errs.append(np.std(global_te) if len(global_te) > 1 else 0.0)
        te_counts.append(len(global_te))

        non_te_centers.append(np.mean(global_non_te) if global_non_te else 0.0)
        non_te_errs.append(np.std(global_non_te) if len(global_non_te) > 1 else 0.0)
        non_counts.append(len(global_non_te))

        tot = len(global_te) + len(global_non_te)
        if tot > 0:
            te_props_list.append(len(global_te) / tot)
            non_te_props_list.append(len(global_non_te) / tot)
        else:
            te_props_list.append(0.0)
            non_te_props_list.append(0.0)

    x = np.arange(len(x_labels))
    width = 0.25
    ekw = dict(elinewidth=0.8, capthick=0.8)

    # Top Subplot: SRE
    b_init = ax1.bar(x - width, init_centers, width, yerr=init_errs, capsize=3,
                     label=r"$\text{Initial States}$", color="mediumseagreen", edgecolor="black", linewidth=0.5, error_kw=ekw)
    b_te = ax1.bar(x, te_centers, width, yerr=te_errs, capsize=3,
                   label=r"$\text{Final TE States}$", color="royalblue", edgecolor="black", linewidth=0.5, error_kw=ekw)
    b_non_te = ax1.bar(x + width, non_te_centers, width, yerr=non_te_errs, capsize=3,
                       label=r"$\text{Final non-TE States}$", color="crimson", edgecolor="black", linewidth=0.5, error_kw=ekw)

    ax1.bar_label(b_init, labels=[f"n={n}" for n in init_counts], fontsize=7, padding=3, color="black")
    ax1.bar_label(b_te, labels=[f"n={n}" for n in te_counts], fontsize=7, padding=5, color="black")
    ax1.bar_label(b_non_te, labels=[f"n={n}" for n in non_counts], fontsize=7, padding=0, color="black")

    ax1.set_ylabel(r"$\text{SRE } (S_2)$", fontsize=11)
    ax1.set_title(f"{tag_str} TE vs non-TE SRE and State Proportions", fontsize=13, fontweight="bold")
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.tick_params(direction="in", top=True, right=True, which="both")
    ax1.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")
    ax1.legend(frameon=True, fontsize=9, loc="upper right")

    # Bottom Subplot: Proportions
    width_prop = 0.35
    ax2.bar(x - width_prop/2, te_props_list, width_prop, label=r"$\text{TE States}$", color="royalblue", edgecolor="black", linewidth=0.5, alpha=0.9)
    ax2.bar(x + width_prop/2, non_te_props_list, width_prop, label=r"$\text{non-TE States}$", color="crimson", edgecolor="black", linewidth=0.5, alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=9.5)
    ax2.set_xlabel(r"$\text{Stabilizer gap } (q)$", fontsize=11)
    ax2.set_ylabel(r"$\text{Proportion of States}$", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.tick_params(direction="in", top=True, right=True, which="both")
    ax2.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")
    ax2.legend(frameon=True, fontsize=9, loc="upper right")

    fig.subplots_adjust(hspace=0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

def main():
    print("Loading dataset files from zip7...")
    loaded_data = load_dataset_files("zip7")
    if not loaded_data:
        print("No .pkl files found in zip7/!")
        return

    tag_str = loaded_data[0][1]['tag']
    print(f"Loaded {len(loaded_data)} data files for dataset tag '{tag_str}' from zip7/.")

    # Config 1: Line chart, all metrics, TE states, Average
    fn1 = f"matplotlib_{tag_str}_config1_te_linechart.png"
    print(f"\n--- Generating Config 1: Linechart ({tag_str}, TE States) ---")
    plot_linechart_config(loaded_data, filter_mode="TE States", output_path=fn1)

    # Config 2: Line chart, all metrics, Non-TE states, Average
    fn2 = f"matplotlib_{tag_str}_config2_non_te_linechart.png"
    print(f"\n--- Generating Config 2: Linechart ({tag_str}, Non-TE States) ---")
    plot_linechart_config(loaded_data, filter_mode="Non-TE States", output_path=fn2)

    # Config 3: TE vs non-TE SRE (2 Stacked Subplots hspace=0)
    fn3 = f"matplotlib_{tag_str}_config3_te_vs_non_te_sre.png"
    print(f"\n--- Generating Config 3: TE vs non-TE SRE ({tag_str}) ---")
    plot_te_vs_non_te_sre_config(loaded_data, output_path=fn3)

if __name__ == "__main__":
    main()
