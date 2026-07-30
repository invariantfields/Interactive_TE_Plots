# Agent Guidelines & Preferences

- **Timestamped Status Updates:** Always include the current local timestamp in every status report/update provided to the user.
- **Post-Push Status Updates:** Automatically provide a detailed status report immediately after every Git auto-push.
- **Estimated Time Remaining (ETR):** Always include estimated time remaining for both current active gaps and full task completions in every status report. When estimating ETR for JAX/GPU tasks, explicitly include upfront JAX JIT compilation overhead (~3.5 min per gap) plus step execution time (~1.5 min per gap), giving ~5 min/gap total. Never base ETR solely on raw step execution speed.
- **Marimo Notebook Verification:** Always run `/home/naga/marimo/bin/marimo check <notebook>.py` after editing any marimo notebook file to guarantee 0 DAG execution errors or cell warnings.
- **Zero-Copy GPU Interop:** Convert JAX GPU tensors to CuPy using `cp.asarray(jax_array)` directly on GPU VRAM instead of routing through CPU host memory (`np.array`).
