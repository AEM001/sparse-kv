from classic.model_classic import SPModel
import torch
import os
import json
import gc
import time
import copy
from datetime import datetime
from accelerate import Accelerator 
from termcolor import colored
import argparse
from classic.edge_cloud import EdgeCloudConfig, EdgeCloudMetrics, NetworkSimulator, RunSettings, synchronize_if_cuda, write_metrics
from shared.kv_cache import initialize_past_key_values
from shared.modeling_llama_kv_target import LlamaForCausalLM as KVLlamaForCausalLM
from transformers import AutoTokenizer

def load_texts_from_jsonl(path: str, max_samples: int = None):
    """
    Load up to `max_samples` texts from a JSONL file.
    Each line must be a JSON object with a "text" key.
    """
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode error at line {i}: {e}")
                continue
            text = obj.get("text")
            if text is None:
                print(f"[Warning] No 'text' field at line {i}, skipping.")
                continue
            texts.append(text)
    return texts


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_local_path(model_id: str) -> str:
    """If model is cached locally, return snapshot path; otherwise return model_id."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    safe_name = model_id.replace("/", "--")
    cache_dir = os.path.join(hf_home, "hub", f"models--{safe_name}")
    snapshots = os.path.join(cache_dir, "snapshots")
    if os.path.isdir(snapshots):
        candidates = []
        fallback = []
        for entry in os.listdir(snapshots):
            snap = os.path.realpath(os.path.join(snapshots, entry))
            if not os.path.isdir(snap):
                continue
            target = fallback if entry == "main" else candidates
            target.append((os.path.getmtime(snap), snap))
        selected = candidates or fallback
        if selected:
            return sorted(selected, reverse=True)[0][1]
    return model_id


def prepare_input_ids(tokenizer, text, device):
    return tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=True
    ).to(device)


@torch.no_grad()
def cloud_ar_generate(base_model, tokenizer, input_ids, settings, edge_cloud_config=None):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    metrics = EdgeCloudMetrics()
    network = NetworkSimulator(edge_cloud_config.network, metrics) if edge_cloud_config else None
    if network is not None:
        network.transfer("uplink", input_ids)

    full_cache_budget = input_ids.shape[1] + settings.max_gen_len + 100
    past_key_values, _, _ = initialize_past_key_values(base_model, full_cache_budget)
    generated = input_ids
    next_token = None
    total_generated = 0
    start_time = datetime.now()

    for step in range(settings.max_gen_len):
        forward_t0 = time.perf_counter()
        if step == 0:
            outputs = base_model(
                input_ids=generated,
                past_key_values=past_key_values,
                use_cache=True,
                return_kv=True,
                init=True,
            )
        else:
            outputs = base_model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_kv=True,
            )
        synchronize_if_cuda(str(generated.device))
        metrics.add("cloud_ar_forward_seconds", time.perf_counter() - forward_t0)

        logits = outputs.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token.to(generated.device)], dim=1)
        total_generated += 1

        if network is not None:
            network.transfer("downlink", next_token)

        if settings.verbose:
            token_str = tokenizer.decode(next_token.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print(colored(token_str, "green"), end=" ", flush=True)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    inference_time = (datetime.now() - start_time).total_seconds()
    tokens_per_sec = round(total_generated / inference_time, 2) if inference_time > 0 else 0.0
    results = {
        "condition": "cloud_ar",
        "total_generated": total_generated,
        "inference_time": inference_time,
        "tokens_per_sec": tokens_per_sec,
        "avg_accept_length": 0.0,
        "accept_length_list": [],
        "cloud_ar": {
            "metrics": metrics.summary(),
        },
    }
    if edge_cloud_config is not None:
        results["cloud_ar"]["config"] = edge_cloud_config.to_dict()
    if settings.output_result_line:
        print(colored(
            f"\n[cloud_ar] Generated {total_generated} tokens in {inference_time:.2f}s. "
            f"Token/sec: {tokens_per_sec}",
            "cyan"
        ))
    return results


def should_use_long_context_guard(settings, texts, tokenizer):
    if not settings.long_context_oom_guard:
        return False
    for text in texts[:1]:
        token_count = len(tokenizer.encode(text, add_special_tokens=True))
        if token_count >= settings.long_context_threshold_tokens:
            return True
    return False


def condition_output_path(metrics_output, condition, sample_idx=None, multi_sample=False):
    if not metrics_output:
        return None
    root, ext = os.path.splitext(metrics_output)
    ext = ext or ".json"
    suffix = f"_{condition}"
    if multi_sample:
        suffix += f"_sample{sample_idx + 1}"
    return f"{root}{suffix}{ext}"

def main():
    parser = argparse.ArgumentParser(
        description="Run SpecExtend inference on a JSONL file of texts."
    )
    parser.add_argument(
        "run_config",
        nargs="?",
        default=None,
        help="Optional JSON run settings file. When set, the file supplies input, model, generation, edge-cloud, network, async, and metrics settings."
    )
    parser.add_argument(
        "--input_file", "-i",
        default=None,
        help="Path to input JSONL file (one JSON obj per line, with a 'text' field)."
    )
    parser.add_argument(
        "--max_samples", "-n",
        type=int, default=None,
        help="Maximum number of samples to read (default: 1)."
    )
    parser.add_argument(
        "--model_name", "-m",
        choices=["vicuna_7b", "longchat_7b"],
        default=None,
        help="Which base model to use (default: vicuna_7b)."
    )
    parser.add_argument(
        "--max_gen_len", "-max",
        type=int, default=None,
        help="Maximum number of tokens to generate(default: 256)."
    )
    parser.add_argument(
        "--use_specextend",
        action="store_true",
        help="Enable SpecExtend speculative decoding (default: False)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from the model."
    )
    parser.add_argument(
        "--output_result_line",
        action="store_true",
        help="If set, print result line-by-line instead of as a block."
    )
    parser.add_argument(
        "--edge_cloud_config",
        type=str,
        default=None,
        help="Path to an edge-cloud JSON config. Enables draft/target split, network simulation, async pipeline, and detailed metrics."
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=None,
        help="Optional path to write per-sample metrics JSON. Overrides metrics_output in the edge-cloud config."
    )
    args = parser.parse_args()

    settings = RunSettings.from_file(args.run_config) if args.run_config else RunSettings()
    if args.input_file is not None:
        settings.input_file = args.input_file
    if args.max_samples is not None:
        settings.max_samples = args.max_samples
    if args.model_name is not None:
        settings.model_name = args.model_name
    if args.max_gen_len is not None:
        settings.max_gen_len = args.max_gen_len
    if args.use_specextend:
        settings.use_specextend = True
    if args.verbose:
        settings.verbose = True
    if args.output_result_line:
        settings.output_result_line = True
    if args.edge_cloud_config:
        settings.edge_cloud = EdgeCloudConfig.from_file(args.edge_cloud_config)
    if args.metrics_output and settings.edge_cloud is not None:
        settings.edge_cloud.metrics_output = args.metrics_output

    if not settings.input_file:
        parser.error("input_file is required either in the run settings file or via --input_file")

    base_model_map = {
        "vicuna_7b":  "lmsys/vicuna-7b-v1.5-16k",
        "longchat_7b": "lmsys/longchat-7b-16k",
    }
    draft_model_map = {
        "vicuna_7b":  "double7/vicuna-68m",
        "longchat_7b": "JackFram/llama-68m",
    }

    base_model_path  = resolve_local_path(base_model_map[settings.model_name])
    draft_model_path = resolve_local_path(draft_model_map[settings.model_name])

    texts = load_texts_from_jsonl(settings.input_file, settings.max_samples)
    if not texts:
        print("No valid texts loaded; exiting.")
        return

    conditions = settings.conditions or [settings.condition]
    valid_conditions = {"cloud_ar", "edge_cloud_sync", "edge_cloud_async"}
    unknown = [condition for condition in conditions if condition not in valid_conditions]
    if unknown:
        parser.error(f"Unknown condition(s): {', '.join(unknown)}. Valid: {', '.join(sorted(valid_conditions))}")

    for condition in conditions:
        clear_cuda()
        condition_settings = copy.deepcopy(settings)
        condition_settings.condition = condition
        condition_edge_cloud = condition_settings.edge_cloud
        if condition == "cloud_ar":
            print(colored(f"\n=== Condition: cloud_ar ===", "yellow"))
            cloud_device = condition_edge_cloud.cloud_device if condition_edge_cloud is not None else "cuda"
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).eval().to(cloud_device)
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
            guarded = should_use_long_context_guard(condition_settings, texts, tokenizer)
            warmup_runs = 0 if guarded else condition_settings.warmup_runs
            if guarded and condition_settings.warmup_runs > 0:
                print(colored("Long-context guard: skipping cloud_ar warmup to reduce 16K OOM risk.", "yellow"))

            for idx, text in enumerate(texts[:1]):
                input_ids = prepare_input_ids(tokenizer, text, cloud_device)
                for _ in range(warmup_runs):
                    warmup_settings = copy.deepcopy(condition_settings)
                    warmup_settings.max_gen_len = 5
                    warmup_settings.output_result_line = False
                    _ = cloud_ar_generate(base_model, tokenizer, input_ids, warmup_settings, condition_edge_cloud)

            for idx, text in enumerate(texts):
                print(colored(f"\n=== Sample {idx+1}/{len(texts)} [{condition}] ===", "yellow"))
                input_ids = prepare_input_ids(tokenizer, text, cloud_device)
                results = cloud_ar_generate(base_model, tokenizer, input_ids, condition_settings, condition_edge_cloud)
                if condition_edge_cloud and condition_edge_cloud.metrics_output:
                    output_path = condition_output_path(condition_edge_cloud.metrics_output, condition, idx, len(texts) > 1)
                    write_metrics(output_path, {
                        "condition": condition,
                        "sample_index": idx,
                        "input_file": condition_settings.input_file,
                        "model_name": condition_settings.model_name,
                        "run_settings": condition_settings.to_dict(),
                        "results": results,
                    })
                    print(colored(f"Wrote metrics to {output_path}", "cyan"))
            del base_model
            clear_cuda()
            continue

        if condition_edge_cloud is None:
            parser.error(f"{condition} requires edge_cloud settings in the run config")

        print(colored(f"\n=== Condition: {condition} ===", "yellow"))
        condition_settings.use_specextend = True
        condition_edge_cloud.async_pipeline.enabled = condition == "edge_cloud_async"

        model = SPModel.from_pretrained(
            base_model_path=base_model_path,
            draft_model_path=draft_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval()
        model.configure_edge_cloud(condition_edge_cloud)
        print(colored(
            f"Edge-cloud mode enabled: draft on {condition_edge_cloud.edge_device}, "
            f"target on {condition_edge_cloud.cloud_device}, async={condition_edge_cloud.async_pipeline.enabled}",
            "cyan"
        ))
        tokenizer = model.tokenizer
        guarded = should_use_long_context_guard(condition_settings, texts, tokenizer)
        warmup_runs = 0 if guarded else condition_settings.warmup_runs
        if guarded and condition_settings.warmup_runs > 0:
            print(colored(f"Long-context guard: skipping {condition} warmup to reduce 16K OOM risk.", "yellow"))
        if guarded and condition == "edge_cloud_async" and condition_edge_cloud.async_pipeline.draft_probe_tokens > 0:
            condition_edge_cloud.async_pipeline.draft_probe_tokens = 0
            print(colored("Long-context guard: disabling async draft probes to reduce 16K OOM risk.", "yellow"))

        print(colored(f'Warming up GPUs for {condition}...', 'yellow'))
        for text in texts[:1]:
            input_ids = prepare_input_ids(tokenizer, text, model._target_device())
            for _ in range(warmup_runs):
                _ = model.spgenerate(
                    input_ids,
                    temperature=0,
                    max_new_tokens=5,
                    output_result_line=False,
                    verbose=False,
                    use_specextend=condition_settings.use_specextend,
                    retrieval_chunk_size=condition_settings.retrieval_chunk_size,
                    retrieve_top_k=condition_settings.retrieve_top_k,
                    retrieve_every_n_steps=condition_settings.retrieve_every_n_steps,
                    retrieval_verbose=condition_settings.retrieval_verbose
                )
        print(colored(f'Warmup complete for {condition}!', 'yellow'))

        for idx, text in enumerate(texts):
            print(colored(f"\n=== Sample {idx+1}/{len(texts)} [{condition}] ===", 'yellow'))
            input_ids = prepare_input_ids(tokenizer, text, model._target_device())
            results = model.spgenerate(
                input_ids,
                temperature=0,
                max_new_tokens=condition_settings.max_gen_len,
                output_result_line=condition_settings.output_result_line,
                verbose=condition_settings.verbose,
                use_specextend=condition_settings.use_specextend,
                retrieval_chunk_size=condition_settings.retrieval_chunk_size,
                retrieve_top_k=condition_settings.retrieve_top_k,
                retrieve_every_n_steps=condition_settings.retrieve_every_n_steps,
                retrieval_verbose=condition_settings.retrieval_verbose
            )
            results["condition"] = condition
            print(colored(
                f"{condition} metrics: {results['total_generated']} tokens, "
                f"{results['inference_time']:.2f}s, {results['tokens_per_sec']} tok/s",
                "cyan"
            ))
            if condition_edge_cloud.metrics_output:
                output_path = condition_output_path(condition_edge_cloud.metrics_output, condition, idx, len(texts) > 1)
                write_metrics(output_path, {
                    "condition": condition,
                    "sample_index": idx,
                    "input_file": condition_settings.input_file,
                    "model_name": condition_settings.model_name,
                    "run_settings": condition_settings.to_dict(),
                    "results": results,
                })
                print(colored(f"Wrote metrics to {output_path}", "cyan"))

        del model
        clear_cuda()

if __name__ == "__main__":
    main()
