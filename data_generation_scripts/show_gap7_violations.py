import pickle
import numpy as np

filepath = "zip2/7_qbt_2000_sds_ptmzng_jfr_1500_stps_7.pkl"
with open(filepath, "rb") as f:
    data = pickle.load(f)

viols = data["total_violation"] # (2000, 31)
viols_mean = np.mean(viols, axis=0)
sre_mean = np.mean(data["sre"], axis=0)

print(f"Gap 7 Total Seeds: {viols.shape[0]}")
print(f"Gap 7 Total Steps: {(viols.shape[1]-1)*50}")
print("\nStep-by-Step Trajectory for Gap 7:")
print(f"{'Step':>6} | {'Mean Violation':>16} | {'Mean SRE':>10}")
print("-" * 40)
for idx, (v, s) in enumerate(zip(viols_mean, sre_mean)):
    step = idx * 50
    print(f"{step:6d} | {v:16.4e} | {s:10.4f}")
