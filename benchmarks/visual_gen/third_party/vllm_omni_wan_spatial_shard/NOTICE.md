# Vendored: vLLM-Omni Wan VAE spatially-sharded decode (benchmark reference only)

Source: https://github.com/vllm-project/vllm-omni, commit `730632e` (Thu 16 Oct 2026),
`vllm_omni/diffusion/distributed/autoencoders/wan_spatial_shard.py` (itself adapted from
SGLang's and FastVideo's spatial-parallel VAE decode, see the file header) plus
`_persistent_input_buffer` from `wan_decoder_utils.py`. License: Apache-2.0 (see LICENSE).
Copied verbatim except for import rewrites (`vllm.logger`, `vllm_omni...wan_decoder_utils`,
`vllm_omni.diffusion.models.interface` -> `_shim.py`). Used only as the multi-GPU vLLM-Omni
comparison backend (`vllm_spatial_shard`) in `benchmarks/visual_gen/bench_wan_vae.py`; vLLM-Omni
does not combine it with its VAE fast path, so the single-GPU vLLM-Omni point stays the fast
path. Not part of the TensorRT-LLM library.
