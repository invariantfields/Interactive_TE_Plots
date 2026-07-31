#!/usr/bin/env python3
import os
import re
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

OUTPUT_DIR = "plots"

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
            'label': f"$q={gap}$",
            'clean_prefix': f"{qbt}q_{sds}_{stps}stps"
        }
    
    m_gap = re.search(r'(\d+)\.pkl$', filename)
    gap_val = int(m_gap.group(1)) if m_gap else 0
    return {
        'qbt': 7,
        'sds': 2000,
        'stps': 2500,
        'gap': gap_val,
        'label': f"$q={gap_val}$",
        'clean_prefix': "7q_2000_2500stps"
    }

def parse_gap_sort_key(filepath):
    info = parse_filename_info(filepath)
    return info['gap']

def get_dynamic_step_mes(file_path, traj_len):
    info = parse_filename_info(file_path)
    if info.get('stps') and traj_len > 1:
        return float(info['stps']) / (traj_len - 1)
    return 50.0

def load_dataset_runs(data_dir="zip7"):
    files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")), key=parse_gap_sort_key)
    run_groups = defaultdict(list)
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        info = parse_filename_info(f)
        run_groups[info['clean_prefix']].append((data, info, f))
    return run_groups

# -------------------------------------------------------------------
# Config 1 & Config 2: Line Chart
# -------------------------------------------------------------------
def plot_linechart_config(loaded_data, filter_mode="TE States", output_path="te_linechart.png"):
    selected_metrics = ["sre", "average_purity", "max_purity"]
    metric_map = {
        "sre": "SRE",
        "average_purity": "Average Purity",
        "max_purity": "Max Purity"
    }
    colors = ["#0052CC", "#FF2A54", "#00875A", "#FFAB00", "#6554C0", "#00B8D9", "#FF5630", "#36B37E"]

    n_plots = len(selected_metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5), sharex=True, squeeze=False, facecolor="white")

    for col_idx, metric in enumerate(selected_metrics):
        ax = axes[0, col_idx]
        ax.set_title(metric_map[metric], fontsize=13)

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
            ax.plot(steps, center, label=info['label'], color=color, linewidth=2.0)
            ax.fill_between(steps, lower_bound, upper_bound, color=color, alpha=0.2, edgecolor="none")

        ax.set_xlabel(r"$\text{Optimization Steps}$", fontsize=11)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(direction="in", top=True, right=True, which="both")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0")
        ax.legend(frameon=True, fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

# -------------------------------------------------------------------
# Config 3: TE vs non-TE SRE
# -------------------------------------------------------------------
def plot_te_vs_non_te_sre_config(loaded_data, output_path="te_vs_non_te_sre.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True, facecolor="white", gridspec_kw={"hspace": 0})
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    x_labels = []
    init_centers, init_errs, init_counts = [], [], []
    te_centers, te_errs, te_counts = [], [], []
    non_te_centers, non_te_errs, non_counts = [], [], []

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
        non_te_errs.append(np.std(global_non_te) if len(global_non_te) > 0 else 0.0)
        non_counts.append(len(global_non_te))

    x = np.arange(len(x_labels))
    width = 0.25
    ekw = dict(elinewidth=0.8, capthick=0.8)

    # Subplot 1 (Top): SRE Values
    ax1.bar(x - width, init_centers, width, yerr=init_errs, capsize=3, label=r"$\text{Initial SRE}$", color="mediumseagreen", edgecolor="black", linewidth=0.5, error_kw=ekw)
    ax1.bar(x, te_centers, width, yerr=te_errs, capsize=3, label=r"$\text{Final TE SRE}$", color="royalblue", edgecolor="black", linewidth=0.5, error_kw=ekw)
    ax1.bar(x + width, non_te_centers, width, yerr=non_te_errs, capsize=3, label=r"$\text{Final non-TE SRE}$", color="crimson", edgecolor="black", linewidth=0.5, error_kw=ekw)
    ax1.set_ylabel(r"$\text{SRE } (S_2)$", fontsize=11)
    ax1.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")
    ax1.legend(frameon=True, fontsize=9, loc="upper right")

    # Subplot 2 (Bottom): Upright Proportions
    te_props_list = []
    non_te_props_list = []
    for tc, nc in zip(te_counts, non_counts):
        tot = tc + nc
        if tot > 0:
            te_props_list.append(tc / tot)
            non_te_props_list.append(nc / tot)
        else:
            te_props_list.append(0.0)
            non_te_props_list.append(0.0)

    ax2.bar(x - width/2, te_props_list, width, label=r"$\text{TE States}$", color="royalblue", edgecolor="black", linewidth=0.5, alpha=0.9)
    ax2.bar(x + width/2, non_te_props_list, width, label=r"$\text{non-TE States}$", color="crimson", edgecolor="black", linewidth=0.5, alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=9.5)
    ax2.set_xlabel(r"$\text{Stabilizer gap } (q)$", fontsize=11)
    ax2.set_ylabel(r"$\text{Proportion of States}$", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax2.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")

    fig.subplots_adjust(hspace=0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

def main():
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "zip7"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading dataset files from {target_dir}...")
    run_groups = load_dataset_runs(target_dir)
    if not run_groups:
        print(f"No .pkl files found in {target_dir}/!")
        return

    print(f"Found {len(run_groups)} distinct run(s) in {target_dir}/:")

    for prefix, loaded_data in run_groups.items():
        print(f"\nProcessing Run: {prefix} ({len(loaded_data)} files)")

        fn1 = os.path.join(OUTPUT_DIR, f"{prefix}_te_linechart.png")
        fn2 = os.path.join(OUTPUT_DIR, f"{prefix}_non_te_linechart.png")
        fn3 = os.path.join(OUTPUT_DIR, f"{prefix}_te_vs_non_te_sre.png")

        print(f"--- Generating Config 1: Linechart ({prefix}, TE States) ---")
        plot_linechart_config(loaded_data, filter_mode="TE States", output_path=fn1)

        print(f"--- Generating Config 2: Linechart ({prefix}, Non-TE States) ---")
        plot_linechart_config(loaded_data, filter_mode="Non-TE States", output_path=fn2)

        print(f"--- Generating Config 3: TE vs non-TE SRE ({prefix}) ---")
        plot_te_vs_non_te_sre_config(loaded_data, output_path=fn3)

if __name__ == "__main__":
    main()
