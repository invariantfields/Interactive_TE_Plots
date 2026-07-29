import time
import numpy as np
import cupy as cp

# 1. Native CuPy GPU Implementation matching Julia HadaMAG's exact unnormalized FHT formula
def create_unnormalized_hadamard_gpu(n_qubits: int) -> cp.ndarray:
    # Unnormalized 1-qubit Hadamard matrix (elements +/- 1)
    H1 = cp.array([[1.0, 1.0], [1.0, -1.0]], dtype=cp.float64)
    H_n = H1
    for _ in range(n_qubits - 1):
        H_n = cp.kron(H_n, H1)
    return H_n

_N_QUBITS_CACHED = None
_H_GPU = None
_XOR_INDICES_GPU = None

def _init_gpu_cache(n_qubits: int = 7):
    global _N_QUBITS_CACHED, _H_GPU, _XOR_INDICES_GPU
    if _N_QUBITS_CACHED != n_qubits:
        dim = 2**n_qubits
        _H_GPU = create_unnormalized_hadamard_gpu(n_qubits)
        x_idx = cp.arange(dim, dtype=cp.int32)[:, None]
        b_idx = cp.arange(dim, dtype=cp.int32)[None, :]
        _XOR_INDICES_GPU = x_idx ^ b_idx
        _N_QUBITS_CACHED = n_qubits

def compute_sre_native_cupy_batch(psi_batch_np: np.ndarray) -> np.ndarray:
    num_starts, dim = psi_batch_np.shape
    n_qubits = int(np.log2(dim))
    _init_gpu_cache(n_qubits)
    
    psi_gpu = cp.array(psi_batch_np, dtype=cp.complex128)
    norms = cp.linalg.norm(psi_gpu, axis=1, keepdims=True)
    norms = cp.where(norms > 1e-12, norms, 1.0)
    psi_gpu = psi_gpu / norms
    
    # Outer product for each state: V[s, x, b] = real(conj(psi[s, x]) * psi[s, x ^ b])
    psi_conj = cp.conj(psi_gpu)[:, :, None]
    psi_xor = psi_gpu[:, _XOR_INDICES_GPU]
    V_batch = cp.real(psi_conj * psi_xor)
    
    # Unnormalized Fast Walsh-Hadamard Transform matching HadaMAG call_fht!
    Xi_batch = cp.matmul(_H_GPU, V_batch)
    
    # HadaMAG formula (Serial.jl line 50): -log2( mSAM / dim )
    xi_4_sum = cp.sum(Xi_batch ** 4, axis=(1, 2))
    sre_batch = -cp.log2(xi_4_sum / dim)
    
    return cp.asnumpy(sre_batch)

# 2. Julia HadaMAG CUDA Reference
def get_julia_handle():
    from juliacall import Main as jl
    jl.seval("using Logging; disable_logging(Logging.Error)")
    jl.seval("using CUDA; using HadaMAG; using LinearAlgebra: norm")
    jl.seval("""
    function jl_compute_sre_vector(psi_batch_np, alpha, n_qubits, dim, num_starts)
        results = zeros(Float64, num_starts)
        for i in 1:num_starts
            psi_row = psi_batch_np[i, :]
            psi_jl = Vector{ComplexF64}(psi_row)
            nrm = norm(psi_jl)
            if nrm > 1e-12
                psi_jl ./= nrm
            end
            psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, Int(n_qubits), Int(dim))
            sre_result, lost_norm = SRE(psi_sv, alpha, backend= :cuda, progress=false, batch=256, threads=512)
            results[i] = sre_result
        end
        return results
    end
    """)
    return jl

if __name__ == "__main__":
    n_qubits = 7
    dim = 128
    num_states = 100
    
    print("=================================================================")
    print(f" STATE-BY-STATE VERIFICATION: Exact HadaMAG Formula Matching")
    print(f" System: {n_qubits} Qubits ({dim} dims) | Total Random States: {num_states}")
    print("=================================================================")

    np.random.seed(42)
    z = np.random.randn(num_states, dim) + 1j * np.random.randn(num_states, dim)
    psi_batch = z / np.linalg.norm(z, axis=1, keepdims=True)

    # 1. Compute Native CuPy GPU SRE
    sre_native = compute_sre_native_cupy_batch(psi_batch)

    # 2. Compute Julia HadaMAG CUDA SRE
    jl = get_julia_handle()
    sre_julia = np.array(jl.jl_compute_sre_vector(psi_batch, 2, n_qubits, dim, num_states))

    # 3. Calculate differences
    abs_diffs = np.abs(sre_native - sre_julia)
    rel_diffs = abs_diffs / np.maximum(np.abs(sre_julia), 1e-12)

    max_abs_diff = np.max(abs_diffs)
    mean_abs_diff = np.mean(abs_diffs)
    r2_score = np.corrcoef(sre_native, sre_julia)[0, 1]**2

    print("\n-----------------------------------------------------------------")
    print(" FIRST 10 STATES INDIVIDUAL COMPARISON:")
    print("-----------------------------------------------------------------")
    print(f"{'State #':<8} | {'Native CuPy SRE':<20} | {'Julia HadaMAG SRE':<20} | {'Abs Difference':<18}")
    print("-" * 75)
    for i in range(10):
        print(f"{i+1:<8} | {sre_native[i]:<20.12f} | {sre_julia[i]:<20.12f} | {abs_diffs[i]:<18.2e}")

    print("\n=================================================================")
    print(" STATISTICAL EQUIVALENCE SUMMARY:")
    print(f" Total States Verified:      {num_states}")
    print(f" Max Absolute Difference:    {max_abs_diff:.4e}")
    print(f" Mean Absolute Difference:   {mean_abs_diff:.4e}")
    print(f" R-squared Correlation (R²): {r2_score:.12f}")
    
    if max_abs_diff < 1e-10:
        print("\n VERDICT: PERFECT MATCH! Native CuPy GPU SRE is mathematically identical to Julia HadaMAG.")
    else:
        print(f"\n VERDICT: Max diff = {max_abs_diff:.2e}")
    print("=================================================================")
