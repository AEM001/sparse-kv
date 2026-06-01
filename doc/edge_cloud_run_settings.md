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
