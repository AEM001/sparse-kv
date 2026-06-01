# Edge-Cloud Run Settings

The edge-cloud runner uses `specextend/configs/edge_cloud.json` as a command settings file. Edit that file first, then run:

```bash
cd specextend
python run_classic.py configs/edge_cloud.json
```

The settings file controls the full command:

- `input_file`: JSONL workload file.
- `max_samples`: number of samples to read from `input_file`.
- `model_name`: `vicuna_7b` or `longchat_7b`.
- `max_gen_len`: maximum generated tokens per sample.
- `use_specextend`: enables SpecExtend speculative decoding.
- `verbose`: prints accepted/resampled tokens while generating.
- `output_result_line`: prints the run summary line.
- `retrieval_chunk_size`, `retrieve_top_k`, `retrieve_every_n_steps`, `retrieval_verbose`: SpecExtend retrieval cache controls.
- `warmup_runs`: number of short warmup generations before measured samples.
- `edge_cloud.edge_device`: GPU for the draft model.
- `edge_cloud.cloud_device`: GPU for the target model.
- `edge_cloud.network.enabled`: enables simulated network delay.
- `edge_cloud.network.rtt_ms`: simulated round-trip time in milliseconds.
- `edge_cloud.network.uplink_mbps`: simulated edge-to-cloud bandwidth.
- `edge_cloud.network.downlink_mbps`: simulated cloud-to-edge bandwidth.
- `edge_cloud.async_pipeline.enabled`: overlaps cloud verification with edge-side draft probes.
- `edge_cloud.async_pipeline.draft_probe_tokens`: speculative probe tokens per polling cycle while waiting for cloud verification.
- `edge_cloud.async_pipeline.poll_interval_ms`: sleep interval between async polling cycles.
- `edge_cloud.metrics_output`: output path for detailed JSON metrics.

The legacy CLI flags still exist for quick overrides, for example:

```bash
python run_classic.py configs/edge_cloud.json --max_gen_len 512 --metrics_output logs/run_512.json
```

For reproducible experiments, prefer putting all parameters in the JSON file and using the one-argument command.

---

## Bug Log — 2026-06-01 11:52

### Issue 1: target model attention mask hard-coded to default GPU (`shared/modeling_llama_kv_target.py`)

**Location:** `shared/modeling_llama_kv_target.py`, lines ~1145–1148 (inside `LlamaModel.forward`)

**Problem:**
- `attention_mask = torch.zeros(...).float().cuda()` creates the mask on the *default* GPU (`cuda:0`), ignoring the actual device of `tree_attention_mask`.
- `torch.tensor(0, dtype=torch.float32)` creates a CPU tensor.
- In edge-cloud mode the target model lives on `cuda:1`; mixing `cuda:0`/CPU tensors with `cuda:1` tensors in `torch.where` triggers `CUDA error: an illegal memory access was encountered` (or `CUBLAS_STATUS_INTERNAL_ERROR` when async CUDA errors are deferred).

**Single-GPU mode masks this because both models are on `cuda:0`.**

---

### Issue 2: async probe `position_ids` out of sync with KV cache (`classic/model_classic.py`)

**Location:** `classic/model_classic.py`, `_edge_cloud_tree_decoding`, `probe_state` construction (lines ~435–440)

**Problem:**
- `probe_state["position_ids"]` is set to `draft_position_ids[-1:]`. That value carries the *tree-decoded* position offset (e.g. `init_len + tree_depth`).
- `probe_state["past_key_values"]` is set to `self.draft_stable_kv`. In edge-cloud mode `draft_stable_kv` is **only updated during the initial `init=True` forward**; it is never refreshed after each accepted tree, so its length stays at the prefill length (e.g. ~1005).
- During `_async_edge_draft_probe`, the draft model receives a `position_ids` value (e.g. 1054) that is **larger than the KV-cache sequence length** (1005). Rotary embedding’s `index_select` then indexes out of bounds, yielding:
  - `CUDA error: device-side assert triggered` (`srcIndex < srcSelectDimSize`)
  - or `RuntimeError: The size of tensor a (1005) must match the size of tensor b (1055)`

**Non-async edge-cloud mode is unaffected because `_async_edge_draft_probe` is never called.**

---

### Issue 3: `resolve_local_path` can return stale `snapshots/main` directory (`run_classic.py`)

**Location:** `run_classic.py`, `resolve_local_path` function

**Problem:**
- The helper returns the first directory it finds under `snapshots/`. If a `main/` symlink directory exists alongside the real commit-hash snapshot, alphabetical order may pick `main/` first.
- Transformers then loads from `snapshots/main`, which may trigger spurious `Some weights were not initialized` warnings for buffers such as `rotary_emb.inv_freq`.
- This does not crash inference, but it clutters logs and is fragile.

---

## Fix Log — 2026-06-01

### Fix 1: target attention mask now follows the target device

Updated `specextend/shared/modeling_llama_kv_target.py` so tree attention masks are allocated on `tree_attention_mask.device` instead of the default CUDA device. The zero scalar used by `torch.where` is also created on the same device. This prevents `cuda:0`/CPU tensors from mixing with a target model placed on `cuda:1`.

### Fix 2: async probe now uses KV-cache-relative positions

Updated `specextend/classic/model_classic.py` so async edge probes initialize `position_ids` from the current draft KV cache length, not from `draft_position_ids[-1:]`, which carries tree-position offsets. The probe also uses a matching local `past_key_position_ids` range. This keeps probe positions and probe KV length aligned while cloud verification is running.

### Fix 3: local snapshot resolution avoids stale `main`

Updated `specextend/run_classic.py` so local Hugging Face snapshot resolution prefers real snapshot directories over a `main` directory/symlink and chooses the newest candidate by modification time. `main` remains a fallback only if no other snapshot directory exists.

---

## Change Log — 2026-06-01

### Three explicit edge-cloud experiment conditions

Updated `specextend/run_classic.py` and `specextend/configs/edge_cloud.json` so one command settings file can run:

- `cloud_ar`: pure cloud autoregressive target request, loading only the target model.
- `edge_cloud_sync`: edge draft plus cloud target with simulated network and synchronous verification.
- `edge_cloud_async`: edge draft plus cloud target with simulated network and async edge draft probes while waiting for cloud verification.

The runner now executes conditions sequentially and clears CUDA cache between them. This avoids keeping the cloud AR target-only baseline and speculative draft/target models resident at the same time.

### 16K OOM guard

Added `long_context_oom_guard` and `long_context_threshold_tokens` settings. When a prompt reaches the threshold, warmup runs are skipped and async draft probes are disabled for `edge_cloud_async`, because those extra probe KV allocations are likely to cause OOM at 16K. The actual requested workload and generation length are unchanged.
