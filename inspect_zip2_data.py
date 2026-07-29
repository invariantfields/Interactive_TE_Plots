import os
import glob
import pickle
import numpy as np

pkl_files = sorted(glob.glob("zip2/*.pkl"))
print(f"Found {len(pkl_files)} pkl files in zip2/")

for fpath in pkl_files:
    print(f"\n--- File: {fpath} ---")
    try:
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            print("Keys in pickle:", list(data.keys()))
            if "sre" in data:
                sre = np.array(data["sre"])
                print("SRE shape:", sre.shape)
                print("SRE mean at step 0:", np.mean(sre[:, 0]))
                if sre.shape[1] > 1:
                    print("SRE mean at step 1:", np.mean(sre[:, 1]))
                    print("SRE mean at step 5:", np.mean(sre[:, 5]) if sre.shape[1] > 5 else "N/A")
                    print("SRE mean at last step:", np.mean(sre[:, -1]))
                    print("Are step 0 and step 1 SRE identical?", np.allclose(sre[:, 0], sre[:, 1]))
                    print("Are step 1 and last step SRE identical?", np.allclose(sre[:, 1], sre[:, -1]))
            if "total_violation" in data:
                viol = np.array(data["total_violation"])
                print("Violation shape:", viol.shape)
                print("Violation mean at step 0:", np.mean(viol[:, 0]))
                if viol.shape[1] > 1:
                    print("Violation mean at step 1:", np.mean(viol[:, 1]))
                    print("Violation mean at last step:", np.mean(viol[:, -1]))
                    print("Are step 0 and step 1 Violation identical?", np.allclose(viol[:, 0], viol[:, 1]))
                    print("Are step 1 and last step Violation identical?", np.allclose(viol[:, 1], viol[:, -1]))
    except Exception as e:
        print(f"Error inspecting {fpath}: {e}")
