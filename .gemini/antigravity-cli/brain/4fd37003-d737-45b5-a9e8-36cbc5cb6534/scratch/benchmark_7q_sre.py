import time
import numpy as np
from juliacall import Main as jl

jl.seval("using Logging")
jl.seval("disable_logging(Logging.Error)")
jl.seval("using HadaMAG")
jl.seval("using LinearAlgebra: norm")

jl.seval("""
function jl_compute_sre_batch(psi_batch_np, alpha, n_qubits, dim, num_starts)
    results = zeros(Float64, num_starts)
    for i in 1:num_starts
        psi_row = psi_batch_np[i, :]
        psi_jl = Vector{ComplexF64}(psi_row)
        nrm = norm(psi_jl)
        if nrm > 1e-12
            psi_jl ./= nrm
        end
        psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, Int(n_qubits), Int(dim))
        sre_result, lost_norm = SRE(psi_sv, alpha, backend= :CUDA, progress=false)
        results[i] = sre_result
    end
    return results
end
""")

dim = 128
n_qubits = 7
num_starts = 200

# Benchmark
psi_batch = np.random.randn(num_starts, dim) + 1j * np.random.randn(num_starts, dim)

t0 = time.time()
res = jl.jl_compute_sre_batch(psi_batch, 2, n_qubits, dim, num_starts)
t1 = time.time()

print(f"7-qubit 200-sample CPU SRE batch time: {t1 - t0:.3f} seconds")
