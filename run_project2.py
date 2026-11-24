"""Evaluate Gemma 3 dense versus sliding-window attention on one A100 80 GB."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import json
import os
from pathlib import Path

import yaml

import config
import llm_util
from inference_timing import TimeCalculationLLMInference
from memory_estimation import MemKind, MemoryEstimator
from util import flush_log_queue


ROOT = Path(__file__).resolve().parent


def single_gpu_peak(tc):
    """Compute the same peak as sequential inference memory replay on one GPU.

    Inference replay retains weights and KV cache, then allocates and releases
    each operation's transient memory immediately. There are no overlapping
    operations on the single device, so its peak is static + max(transient).
    Capacity searches therefore do not need to repeat expensive GEMM pricing.
    """
    if tc.use_moe or any(x != 1 for x in (tc.tp, tc.cp, tc.pp, tc.mb)):
        raise ValueError("The fast capacity search requires one GPU, one microbatch, and a dense model.")
    estimator = MemoryEstimator(tc)
    prefill = tc.seq_len - tc.model.decode_len
    phases = [(prefill, prefill, None)] if prefill > 0 else []
    if tc.model.decode_len:
        shapes = llm_util.process_decode_gemm_shapes(
            tc, batch_size=tc.batch_size, current_seq_len=tc.seq_len,
            d_model=tc.hidden_dim, num_heads=tc.num_heads, kv_heads=tc.kv_heads,
            intermediate_size=tc.intermediate_size, vocab_size=tc.vocab_size,
            model_type=tc.model_type, query_tokens=1,
        )
        phases.append((1, tc.seq_len, shapes))
    peaks = []
    for query_length, kv_tokens, shapes in phases:
        data = estimator.build_memory_data(
            mode="inference", batch_size=tc.batch_size, seq_len=query_length,
            kv_cache_tokens=kv_tokens, gemm_shapes=shapes,
        )
        cache = data.get("kv_cache_bytes_by_layer")
        cache_bytes = sum(cache) if cache is not None else data["kv_cache_bytes_per_layer"] * tc.num_layers
        static = (
            data["weight_mem_per_layer_dense"] * tc.num_layers
            + cache_bytes + data["extra_static_bytes_per_device"].get(0, 0.0)
        )
        # TRANSFORMER is the unflattened fallback, not an executed operation.
        transient = max(
            (size for kind, size in data["transient_bytes_by_kind_dense"].items()
             if kind != MemKind.TRANSFORMER), default=0.0,
        )
        peaks.append((static + transient) / 1024**3)
    return max(peaks, default=0.0)


def evaluate(source, hardware, folder, variant, output_tokens, batch, memory_only=False,
             *, prompt_tokens=8192, window_size=None):
    """Keep each run's configs, logs, and generated files in its own directory."""
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if variant not in {"dense", "sliding"}:
        raise ValueError(f"Unknown attention variant: {variant}")
    data = deepcopy(source)
    model = data["model_param"]
    model.update(global_batch_size=batch, seq_len=prompt_tokens + output_tokens, decode_len=output_tokens)
    if variant == "dense":
        model["attention"].pop("window", None)
    elif window_size is not None:
        model["attention"]["window"]["window_size"] = window_size
    model_path = folder / "model.yaml"
    model_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    hw = config.parse_config(str(hardware), config_type="hardware")
    cfg = config.parse_config(str(model_path), config_type="LLM")
    config.validate_configs(hw, cfg)

    # DecodeGraph also writes relative output/LLM paths. Isolate those along
    # with the explicit output directory, preserving the user's prior runs.
    previous_cwd = Path.cwd()
    try:
        os.chdir(folder)
        with (folder / "run.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            tc = TimeCalculationLLMInference(hw, cfg, "LLM", output_dir=str(folder / "output/LLM"))
            if memory_only:
                peak = single_gpu_peak(tc)
                fits = peak <= tc.DRAM.size / 1024**3
                timing = {}
            else:
                timing = tc.calc_total_inference_time()
                peak = tc.memory_peak_gb
                fits = not tc.memory_capacity_exceeded
            flush_log_queue()
    finally:
        os.chdir(previous_cwd)

    row = {
        "attention": variant,
        "output_tokens": output_tokens,
        "batch_size": batch,
        "prompt_tokens": prompt_tokens,
        "window_size": model["attention"].get("window", {}).get("window_size"),
        "prefill_s": timing.get("prefill_time"),
        "decode_s": timing.get("decode_time"),
        "tpot_ms": timing["decode_time"] * 1000 / output_tokens if timing else None,
        "peak_gib": peak,
        "prefill_peak_gib": None if memory_only else tc.prefill_peak_gb,
        "decode_peak_gib": None if memory_only else tc.decode_peak_gb,
        "capacity_gib": tc.DRAM.size / 1024**3,
        "fits": fits,
    }
    (folder / "results.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def largest_batch(source, hardware, folder, variant, tokens):
    """Bracket capacity, then find the last fitting integer using full precision."""
    def fits(batch):
        return evaluate(
            source, hardware, folder / f"batch_{batch}", variant, tokens, batch, memory_only=True
        )["fits"]

    low, high = 0, 1
    while fits(high):
        low, high = high, high * 2
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--find-max-batch", action="store_true", help="Also search maximum fitting batch sizes")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/project2/evaluation")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    hardware = ROOT / "configs/hardware-config/course_project/a100_80GB_1gpu.yaml"
    source_path = ROOT / "configs/model-config/course_project/gemma_3_4b_inf.yaml"
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    rows = []
    for tokens in (128, 256):
        for variant in ("dense", "sliding"):
            print(f"Running {variant}, output={tokens}, batch=16 ...", flush=True)
            row = evaluate(source, hardware, output / f"{variant}_{tokens}", variant, tokens, 16)
            row["max_batch"] = None
            if args.find_max_batch:
                print("  Searching maximum batch ...", flush=True)
                row["max_batch"] = largest_batch(
                    source, hardware, output / "capacity" / f"{variant}_{tokens}", variant, tokens
                )
            rows.append(row)
            print(
                f"  prefill={row['prefill_s']:.4f}s, TPOT={row['tpot_ms']:.4f}ms, "
                f"peak={row['peak_gib']:.4f} GiB, fits={row['fits']}, max_batch={row['max_batch']}",
                flush=True,
            )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
