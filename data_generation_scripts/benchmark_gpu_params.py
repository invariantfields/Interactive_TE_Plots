import os
import time
import numpy as np

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['LD_LIBRARY_PATH'] = '/home/naga/marimo/lib/python3.14/site-packages/nvidia/cu13/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

print("Initializing Julia/CUDA/HadaMAG benchmark...")
from juliacall import Main as jl

jl.seval("using Logging; disable_logging(Logging.Error)")
jl.seval("using CUDA")
jl.seval("using HadaMAG")
jl.seval("using LinearAlgebra: norm")

dev_name = jl.seval("CUDA.name(CUDA.device())")
sm_count = jl.seval("CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)")
max_threads = jl.seval("CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)")

print(f"\n=======================================================")
print(f"GPU Device: {dev_name}")
print(f"Multiprocessors (SMs): {sm_count} | Max Threads/Block: {max_threads}")
print(f"=======================================================\n")

jl.seval("""
function bench_sre(psi_np, alpha, n_qubits, dim, b_size, t_size)
    psi_jl = Vector{ComplexF64}(psi_np)
    nrm = norm(psi_jl)
    if nrm > 1e-12
        psi_jl ./= nrm
    end
    psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, Int(n_qubits), Int(dim))
    t0 = time()
    sre_res, lost_nrm = SRE(psi_sv, alpha, backend= :cuda, progress=false, batch=Int(b_size), threads=Int(t_size))
    t1 = time()
    return sre_res, (t1 - t0)
end
""")

# Generate 5 test statevectors for 7 qubits (dim=128)
np.random.seed(42)
test_states = [np.random.randn(128) + 1j * np.random.randn(128) for _ in range(5)]
for i in range(len(test_states)):
    test_states[i] /= np.linalg.norm(test_states[i])

# Warmup run
jl.bench_sre(test_states[0], 2, 7, 128, 128, 128)

batches = [64, 128, 256, 512, 1024]
threads = [64, 128, 256, 512]

results = []

for b in batches:
    for t in threads:
        try:
            times = []
            for psi in test_states:
                res, elapsed = jl.bench_sre(psi, 2, 7, 128, b, t)
                times.append(elapsed)
            avg_time_ms = np.mean(times) * 1000
            results.append((avg_time_ms, b, t))
            print(f"batch={b:4d} | threads={t:3d} => Avg Time: {avg_time_ms:6.2f} ms")
        except Exception as e:
            print(f"batch={b:4d} | threads={t:3d} => Error: {e}")

results.sort(key=lambda x: x[0])
best_time, best_b, best_t = results[0]

print(f"\n=======================================================")
print(f"OPTIMAL PARAMETERS FOUND:")
print(f"  Best Batch Size: {best_b}")
print(f"  Best Thread Count: {best_t}")
print(f"  Fastest Avg Execution Time: {best_time:.2f} ms per state")
print(f"=======================================================")
