import time
import numpy as np
import cupy as cp

# 1. Native CuPy GPU Implementation
def create_hadamard_gpu(n_qubits: int) -> cp.ndarray:
    H = cp.array([[1.0, 1.0], [1.0, -1.0]], dtype=cp.float64)
    H_n = H
    for _ in range(n_qubits - 1):
        H_n = cp.kron(H_n, H)
    return H_n

_N_QUBITS_CACHED = None
_H_GPU = None
_XOR_INDICES_GPU = None

def _init_gpu_cache(n_qubits: int = 7):
    global _N_QUBITS_CACHED, _H_GPU, _XOR_INDICES_GPU
    if _N_QUBITS_CACHED != n_qubits:
        dim = 2**n_qubits
        _H_GPU = create_hadamard_gpu(n_qubits)
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
    
    psi_conj = cp.conj(psi_gpu)[:, :, None]
    psi_xor = psi_gpu[:, _XOR_INDICES_GPU]
    V_batch = cp.real(psi_conj * psi_xor)
    
    Xi_batch = cp.matmul(_H_GPU, V_batch)
    xi_4_sum = cp.sum(Xi_batch ** 4, axis=(1, 2))
    
    sre_batch = -cp.log2(xi_4_sum / (dim**2))
    return cp.asnumpy(sre_batch)

# 2. Julia HadaMAG CUDA Reference
def get_julia_handle():
    from juliacall import Main as jl
    jl.seval("using Logging; disable_logging(Logging.Error)")
    jl.seval("using CUDA; using HadaMAG; using LinearAlgebra: norm")
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
            sre_result, lost_norm = SRE(psi_sv, alpha, backend= :cuda, progress=false, batch=256, threads=512)
            results[i] = sre_result
        end
        GC.gc()
        return results
    end
    """)
    return jl

if __name__ == "__main__":
    n_qubits = 7
    dim = 2**n_qubits
    num_starts = 50
    num_runs = 20

    print("=================================================================")
    print(f" BENCHMARK: Execution Time Mean & Std Dev ({num_runs} Runs)")
    print(f" System: {n_qubits} Qubits | Batch Size: {num_starts} States")
    print("=================================================================")

    np.random.seed(42)
    z = np.random.randn(num_starts, dim) + 1j * np.random.randn(num_starts, dim)
    psi_batch = z / np.linalg.norm(z, axis=1, keepdims=True)

    # Warmup CuPy
    _ = compute_sre_native_cupy_batch(psi_batch)
    cp.cuda.Stream.null.synchronize()

    native_times = []
    for r in range(num_runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _ = compute_sre_native_cupy_batch(psi_batch)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        native_times.append(t1 - t0)

    native_mean = float(np.mean(native_times))
    native_std = float(np.std(native_times))

    print(f"\n[Native CuPy GPU ({num_starts} states)]")
    print(f"  Mean Time: {native_mean * 1000:.3f} ms ± {native_std * 1000:.3f} ms")
    print(f"  Per State: {(native_mean / num_starts) * 1000:.3f} ms ± {(native_std / num_starts) * 1000:.3f} ms")

    # Warmup Julia
    jl = get_julia_handle()
    _ = jl.jl_compute_sre_batch(psi_batch, 2, n_qubits, dim, num_starts)

    julia_times = []
    for r in range(num_runs):
        t0 = time.perf_counter()
        _ = jl.jl_compute_sre_batch(psi_batch, 2, n_qubits, dim, num_starts)
        t1 = time.perf_counter()
        julia_times.append(t1 - t0)

    julia_mean = float(np.mean(julia_times))
    julia_std = float(np.std(julia_times))

    print(f"\n[Julia HadaMAG CUDA ({num_starts} states)]")
    print(f"  Mean Time: {julia_mean * 1000:.3f} ms ± {julia_std * 1000:.3f} ms")
    print(f"  Per State: {(julia_mean / num_starts) * 1000:.3f} ms ± {(julia_std / num_starts) * 1000:.3f} ms")

    print("\n=================================================================")
    print(f" SUMMARY COMPARISON ({num_runs} TRIALS):")
    print(f" Native CuPy GPU:   {native_mean*1000:.2f} ms ± {native_std*1000:.2f} ms")
    print(f" Julia HadaMAG:     {julia_mean*1000:.2f} ms ± {julia_std*1000:.2f} ms")
    print(f" Speedup Factor:    {(julia_mean / native_mean):.1f}x FASTER")
    print("=================================================================")
