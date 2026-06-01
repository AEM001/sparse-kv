from classic.model_classic import SPModel
import torch
import os
import json
from accelerate import Accelerator 
from termcolor import colored
import argparse
from classic.edge_cloud import EdgeCloudConfig, write_metrics

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

def main():
    parser = argparse.ArgumentParser(
        description="Run SpecExtend inference on a JSONL file of texts."
    )
    parser.add_argument(
        "--input_file", "-i",
        required=True,
        help="Path to input JSONL file (one JSON obj per line, with a 'text' field)."
    )
    parser.add_argument(
        "--max_samples", "-n",
        type=int, default=1,
        help="Maximum number of samples to read (default: 1)."
    )
    parser.add_argument(
        "--model_name", "-m",
        choices=["vicuna_7b", "longchat_7b"],
        default="vicuna_7b",
        help="Which base model to use (default: vicuna_7b)."
    )
    parser.add_argument(
        "--max_gen_len", "-max",
        type=int, default=256,
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

    def resolve_local_path(model_id: str) -> str:
        """If model is cached locally, return snapshot path; otherwise return model_id."""
        import os
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        safe_name = model_id.replace("/", "--")
        cache_dir = os.path.join(hf_home, "hub", f"models--{safe_name}")
        snapshots = os.path.join(cache_dir, "snapshots")
        if os.path.isdir(snapshots):
            for entry in os.listdir(snapshots):
                snap = os.path.join(snapshots, entry)
                if os.path.isdir(snap):
                    return snap
        return model_id

    base_model_map = {
        "vicuna_7b":  "lmsys/vicuna-7b-v1.5-16k",
        "longchat_7b": "lmsys/longchat-7b-16k",
    }
    draft_model_map = {
        "vicuna_7b":  "double7/vicuna-68m",
        "longchat_7b": "JackFram/llama-68m",
    }

    base_model_path  = resolve_local_path(base_model_map[args.model_name])
    draft_model_path = resolve_local_path(draft_model_map[args.model_name])

    texts = load_texts_from_jsonl(args.input_file, args.max_samples)
    if not texts:
        print("No valid texts loaded; exiting.")
        return

    edge_cloud_config = EdgeCloudConfig.from_file(args.edge_cloud_config) if args.edge_cloud_config else None
    if args.metrics_output and edge_cloud_config is not None:
        edge_cloud_config.metrics_output = args.metrics_output

    model = SPModel.from_pretrained(
        base_model_path=base_model_path,
        draft_model_path=draft_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    if edge_cloud_config is not None:
        model.configure_edge_cloud(edge_cloud_config)
        print(colored(
            f"Edge-cloud mode enabled: draft on {edge_cloud_config.edge_device}, "
            f"target on {edge_cloud_config.cloud_device}",
            "cyan"
        ))
    else:
        model = model.cuda()
    tokenizer = model.tokenizer

    accelerator = Accelerator()
    if edge_cloud_config is None:
        model, tokenizer = accelerator.prepare(model, tokenizer)

    # Warmup GPUs
    print(colored(f'Warming up GPUs...', 'yellow'))
    for idx, text in enumerate(texts[:1]):
        input_ids = tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True
        )
        if edge_cloud_config is not None:
            input_ids = input_ids.to(model._target_device())
        else:
            input_ids = input_ids.to(accelerator.device)

        for _ in range(3):
            _ = model.spgenerate(
                input_ids,
                temperature=0,
                max_new_tokens=5,
                output_result_line=False,
                verbose=False,
                use_specextend=args.use_specextend,
                retrieval_chunk_size=32,
                retrieve_top_k=32,
                retrieve_every_n_steps=4,
                retrieval_verbose=False
            )
    print(colored(f'Warmup complete!', 'yellow'))

    for idx, text in enumerate(texts):
        print(colored(f"\n=== Sample {idx+1}/{len(texts)} ===", 'yellow'))
        input_ids = tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True
        )
        if edge_cloud_config is not None:
            input_ids = input_ids.to(model._target_device())
        else:
            input_ids = input_ids.to(accelerator.device)

        results = model.spgenerate(
            input_ids,
            temperature=0,
            max_new_tokens=args.max_gen_len,
            output_result_line=args.output_result_line,
            verbose=args.verbose,
            use_specextend=args.use_specextend,
            retrieval_chunk_size=32,
            retrieve_top_k=32,
            retrieve_every_n_steps=4,
            retrieval_verbose=False
        )
        if edge_cloud_config is not None:
            print(colored(
                f"Edge-cloud metrics: {results['total_generated']} tokens, "
                f"{results['inference_time']:.2f}s, {results['tokens_per_sec']} tok/s",
                "cyan"
            ))
            if edge_cloud_config.metrics_output:
                payload = {
                    "sample_index": idx,
                    "input_file": args.input_file,
                    "model_name": args.model_name,
                    "results": results,
                }
                output_path = edge_cloud_config.metrics_output
                if len(texts) > 1:
                    root, ext = os.path.splitext(output_path)
                    output_path = f"{root}_sample{idx + 1}{ext or '.json'}"
                write_metrics(output_path, payload)
                print(colored(f"Wrote metrics to {output_path}", "cyan"))

if __name__ == "__main__":
    main()
