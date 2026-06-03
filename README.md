<h1 align="center">
  Long-context edge-cloud collaborative speculative decoding
</h1>


## Introduction

While speculative decoding has emerged as an effective, lossless solution to accelerating LLM inference, its performance degrades significantly for even moderately long inputs. This is due to:

1. **Increased latency** in both drafting and verification steps due to the quadratic complexity of standard attention.
2. **Reduced draft accuracy**, as the draft model is typically smaller and trained only on short sequences.

**SpecExtend** addresses this by:

* **Accelerating forward passes** of both the draft and target models, integrating efficient attention mechanisms across all stages (FlashAttention & Hybrid Tree Attention).
* **Introducing Cross-model Retrieval**, a novel cache update strategy that uses the target model's attention scores to dynamically update the draft model’s KV cache with globally relevant context. This allows fine-grained alignment between the target and draft model, boosting both draft speed and accuracy on long inputs without retraining.

SpecExtend achieves up to:

* **2.84×** speedup on the long document summarization task with Vicuna 7B and 68M on inputs up to 16K tokens of GovReport.

SpecExtend also preserves performance on short sequences, is training-free and compatible with SOTA speculative decoding frameworks.

<div align="center">
  <!-- First GIF + italic caption -->
  <figure style="display: inline-block; margin: 0;">
    <img src="figs/2K.gif" alt="2K-token example" style="max-width: 100%;" />
    <figcaption><em>Input length: 2K</em></figcaption>
  </figure>
  <br/><br/>

  <!-- Second GIF + italic caption -->
  <figure style="display: inline-block; margin: 0;">
    <img src="figs/16K.gif" alt="16K-token example" style="max-width: 100%;" />
    <figcaption><em>Input length: 16K</em></figcaption>
  </figure>
  <br/>
</div>

<p align="center">
  Inference is conducted using Vicuna 7B and 68M as target and draft models, on a single A100 80GB GPU at fp16 precision.
</p>


## Installation

### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/jycha98/SpecExtend.git
cd SpecExtend/specextend
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch (CUDA 12.1)

```bash
pip install --upgrade pip
pip install torch==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu121
```

### 3. Install flash-attn (Pre-built Wheel, No Compilation)

Instead of building from source, use the pre-built wheel for faster installation:

```bash
# Download pre-built wheel (flash-attn 2.6.3 + CUDA 12.1 + torch 2.4 + Python 3.12)
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.2/flash_attn-2.6.3%2Bcu121torch2.4-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.6.3+cu121torch2.4-cp312-cp312-linux_x86_64.whl
```

### 4. Install Remaining Dependencies

```bash
pip install transformers==4.41.0 accelerate==0.21.0 sentencepiece==0.1.99 \
    termcolor==3.1.0 tqdm==4.67.1 gradio==5.32.1 ninja==1.11.1.4 \
    protobuf==3.19.0 tokenizers==0.19.1
```

### 5. Download Models

Models are cached under a **data disk** (e.g. `/root/autodl-tmp/huggingface`) to avoid filling up system storage.

```bash
# Option A: Use HuggingFace mirror (no proxy needed)
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download lmsys/vicuna-7b-v1.5-16k
huggingface-cli download double7/vicuna-68m

# Option B: Use proxy acceleration
source /etc/network_turbo
huggingface-cli download lmsys/vicuna-7b-v1.5-16k
huggingface-cli download double7/vicuna-68m
```

**Required models:**
- **Target**: `lmsys/vicuna-7b-v1.5-16k` (~13.5 GB)
- **Draft**: `double7/vicuna-68m` (~260 MB)

### 6. Offline Inference

Before running, set `HF_HOME` so the script loads models from the local cache (no internet required):

```bash
source .venv/bin/activate
export HF_HOME=/root/autodl-tmp/huggingface
```

#### Baseline (without SpecExtend)

```bash
python run_classic.py \
  --input_file data/govreport/govreport_2K.jsonl \
  --model_name vicuna_7b \
  --max_gen_len 256 \
  --max_samples 1
```

#### With SpecExtend

```bash
python run_classic.py \
  --input_file data/govreport/govreport_2K.jsonl \
  --model_name vicuna_7b \
  --use_specextend \
  --verbose \
  --output_result_line \
  --max_gen_len 256 \
  --max_samples 1
```

#### Edge-Cloud Collaborative SpecExtend

Use this mode to compare three deployment conditions from one settings file:

- `cloud_ar`: pure cloud autoregressive target-model request; only the target model is loaded.
- `edge_cloud_sync`: edge draft model plus cloud target model with network simulation, no async edge probing while verification is pending.
- `edge_cloud_async`: edge draft model plus cloud target model with network simulation and async edge-side draft probes while waiting for cloud verification.

The runner executes conditions sequentially to avoid keeping unnecessary models in memory. This is important for 16K contexts, where loading cloud AR and draft/target speculative models at the same time can OOM.

Edit `configs/edge_cloud.json` to change the whole run. This file is a command settings file, not only a network config: it contains the input file, model, generation length, SpecExtend retrieval settings, edge/cloud devices, network condition, async behavior, and metrics output path.

- `conditions`: list of conditions to run, usually `cloud_ar`, `edge_cloud_sync`, and `edge_cloud_async`
- `input_file`, `max_samples`, `model_name`, `max_gen_len`: workload and generation settings
- `use_specextend`, `retrieval_chunk_size`, `retrieve_top_k`, `retrieve_every_n_steps`: SpecExtend settings
- `long_context_oom_guard`: skips warmup and disables async probe KV growth for prompts at or above `long_context_threshold_tokens`
- `edge_cloud.edge_device`: GPU for the draft model, for example `cuda:0`
- `edge_cloud.cloud_device`: GPU for the target model, for example `cuda:1`
- `edge_cloud.network.rtt_ms`: simulated round-trip time
- `edge_cloud.network.uplink_mbps`: edge-to-cloud bandwidth
- `edge_cloud.network.downlink_mbps`: cloud-to-edge bandwidth
- `edge_cloud.async_pipeline.enabled`: whether the edge continues drafting while cloud verification is pending
- `edge_cloud.async_pipeline.draft_probe_tokens`: how many speculative probe tokens the edge attempts per async polling cycle
- `edge_cloud.metrics_output`: JSON metrics output path

```bash
python run_classic.py configs/edge_cloud.json
```

CLI flags still work for quick overrides, but the intended edge-cloud workflow is to edit `configs/edge_cloud.json` and run the simple command above.

The metrics JSON includes generated-token throughput, acceptance lengths, simulated uplink/downlink bytes and seconds, cloud AR forward time, target tree verification time, edge draft time, and async draft probe work. Edge-cloud speculative modes use a cloud-selected chunk protocol: uplink sends the draft tree plus chunk metadata on retrieval steps, and downlink sends only accepted-token metadata plus selected chunk IDs. Metrics are written with the condition name in the filename, for example `edge_cloud_metrics_cloud_ar.json`.

### 7. Performance Comparison (Example)

On a single RTX 3090 with Vicuna 7B + Vicuna 68M, `govreport_2K` sample, generating 256 tokens:

| Method | Wall-clock Time |
|--------|-----------------|
| Baseline (no SpecExtend) | ~15.7 s |
| SpecExtend | ~16.3 s |

> **Note:** Speculative decoding overhead can exceed gains on very short inputs. Significant speedups (up to **2.84x**) are observed on longer sequences (e.g. 16K tokens) where draft model acceptance rates are higher.

## Evaluation

We also provide scripts to evaluate SpecExtend's performance on GovReport and PG-19.

```bash
python eval_classic.py \
  --data_dir data/govreport \
  --samples_per_length 20 \
  --runs_per_sample 2 \
  --model_name vicuna_7b \
  --use_specextend \
  --max_gen_len 256 \
  --output_file eval_results_classic.json
```

## Project Files

| File | Description |
|------|-------------|
| `download_models_mirror.sh` | Download models via HF mirror |
| `monitor_download.py` | Monitor download progress & detect stuck transfers |
| `run_classic.py` | Inference script (modified for local cache paths) |
| `classic/model_classic.py` | Model wrapper (with `local_files_only=True` for offline loading) |
