# Edge-Cloud Conditions

`specextend/configs/edge_cloud.json` is a command settings file. Edit it, then run:

```bash
cd specextend
python run_classic.py configs/edge_cloud.json
```

## Conditions

The `conditions` field controls which deployment conditions run:

- `cloud_ar`: pure cloud autoregressive request. The runner loads only the target model on `edge_cloud.cloud_device`. It does not load the draft model, which keeps this baseline separate and reduces memory pressure.
- `edge_cloud_sync`: draft model on `edge_cloud.edge_device`, target model on `edge_cloud.cloud_device`, simulated network transfer, synchronous verification.
- `edge_cloud_async`: same edge/cloud split and network simulation, but the edge continues async draft probes while waiting for cloud verification.

The default settings run all three conditions sequentially:

```json
"conditions": [
  "cloud_ar",
  "edge_cloud_sync",
  "edge_cloud_async"
]
```

Sequential execution is deliberate: each condition releases CUDA memory before the next condition loads its models.

## 16K OOM Guard

16K prompts are memory-sensitive. The settings file includes:

```json
"long_context_oom_guard": true,
"long_context_threshold_tokens": 16000
```

When the first sample reaches this threshold, the runner:

- skips warmup runs for that condition;
- disables async draft probes for `edge_cloud_async` to avoid growing extra probe KV cache while cloud verification is still running;
- still records the condition and metrics, so the run remains comparable.

This guard does not reduce `max_gen_len` or change the actual requested workload. It only removes extra measurement overhead that is likely to trigger OOM at 16K.

## Metrics

Metrics are written per condition. If `edge_cloud.metrics_output` is `edge_cloud_metrics.json`, outputs look like:

- `edge_cloud_metrics_cloud_ar.json`
- `edge_cloud_metrics_edge_cloud_sync.json`
- `edge_cloud_metrics_edge_cloud_async.json`

For multiple samples, `_sampleN` is appended before `.json`.
