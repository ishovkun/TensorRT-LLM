# Vendored: vLLM-Omni Wan VAE fast path (benchmark reference only)

Source: https://github.com/vllm-project/vllm-omni, commit `730632e` (Thu 15 Oct 2026),
`vllm_omni/diffusion/distributed/autoencoders/wan_vae_fastpath/` plus
`vllm_omni/diffusion/layers/vae_norm_epilogue.py` and `vae_rms_norm.py`.
License: Apache-2.0 (see LICENSE). Copied verbatim except for import rewrites
(`vllm.logger`, `vllm.triton_utils`, `vllm_omni.diffusion.layers` -> local
modules; see `_vllm_shim.py`). Used only as a comparison backend in
`benchmarks/visual_gen/bench_wan_vae.py`; not part of the TensorRT-LLM library.
