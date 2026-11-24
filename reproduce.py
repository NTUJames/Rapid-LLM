"""Reproduce Project 2's required evaluation, two case studies, tables and figures."""

import argparse
from copy import deepcopy
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import tempfile
import time
from types import SimpleNamespace

import yaml

import config
from run_project2 import ROOT, evaluate, largest_batch, single_gpu_peak


MODEL = ROOT / "configs/model-config/course_project/gemma_3_4b_inf.yaml"
HARDWARE = ROOT / "configs/hardware-config/course_project/a100_80GB_1gpu.yaml"
PROMPTS = (512, 1024, 2048, 4096, 8192, 16384)
WINDOWS = (256, 512, 1024, 2048, 4096, 16384)


def verify():
    """Small deterministic checks, without downloading models or needing a GPU."""
    from inference_timing import TimeCalculationLLMInference
    from simulate_inference_graph import DecodeGraph
    from train_timing import TimeCalculationLLM

    source = yaml.safe_load(MODEL.read_text(encoding="utf-8"))
    hw = config.parse_config(str(HARDWARE), config_type="hardware")
    cfg = config.parse_config(str(MODEL), config_type="LLM")
    window = cfg.model_config.attention.window
    assert sum(window.is_global_layer(i, 34) for i in range(34)) == 5
    assert replace(window, last_layer_global=True).is_global_layer(33, 34)

    # Count active tiles independently by enumerating causal token pairs.
    fake = SimpleNamespace(
        cp=1, head_dim=256, attention_tile_size=128, mode="LLM",
        precision=SimpleNamespace(activations=2), num_levels=4, tp=1, O=0,
        roofline=lambda *a, **k: 0, get_gemm_time=lambda *a, **k: (1,),
    )
    for length in (1, 127, 128, 129, 257):
        for width in (1, 127, 128, 129, 1024):
            touched = {(q // 128, k // 128) for q in range(length)
                       for k in range(max(0, q - width + 1), q + 1)}
            actual = TimeCalculationLLM.flash_attention_kernel_forward(
                fake, 1, 256, length, 1, 1, 1, window_size=width
            )[0]
            assert actual == 2 * len(touched), (length, width)

    graph = object.__new__(DecodeGraph)
    graph.config = SimpleNamespace(seq_len=1020, decode_len=10, num_layers=34, sample_every=2**31-1)
    graph.model_config = cfg
    assert {0, 2, 3, 4, 9}.issubset(graph._generate_sample_points())

    # Preserve Gemma's architecture type while shrinking dimensions for a quick check.
    small = deepcopy(source)
    small["model_param"].update(hidden_dim=256, intermediate_size=512, num_layers=6, vocab_size=512)
    small["model_param"]["attention"].update(num_heads=4, kv_heads=2, head_dim=64)
    small["model_param"]["moe"].update(first_k_dense_replace=6, moe_intermediate_size=512)
    with tempfile.TemporaryDirectory(prefix="rapid-project2-check-") as temp:
        temp = Path(temp)

        def run(label, data, variant, prompt, width):
            folder = temp / label
            row = evaluate(data, HARDWARE, folder, variant, 8, 2,
                           prompt_tokens=prompt, window_size=width)
            parsed = config.parse_config(str(folder / "model.yaml"), config_type="LLM")
            tc = TimeCalculationLLMInference(hw, parsed, "LLM", output_dir=str(folder / "capacity"))
            assert row["peak_gib"] == single_gpu_peak(tc), label
            return row

        def equal(left, right):
            for field in ("prefill_s", "decode_s", "peak_gib"):
                assert math.isclose(left[field], right[field], rel_tol=1e-12), field

        equal(run("short-dense", small, "dense", 32, None),
              run("short-local", small, "sliding", 32, 1024))
        dense = run("long-dense", small, "dense", 512, None)
        global_source = deepcopy(small)
        global_source["model_param"]["attention"]["window"]["local_global_interval"] = 1
        equal(dense, run("all-global", global_source, "sliding", 512, 16))
        equal(dense, run("wide-window", small, "sliding", 512, 1024))
        local = run("long-local", small, "sliding", 512, 16)
        assert all(local[k] < dense[k] for k in ("prefill_s", "decode_s", "peak_gib"))

    print("Verification passed: layer pattern, 25 tile boundaries, sampling knee, "
          "dense equivalence, local savings, and capacity/replay agreement.", flush=True)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(output):
    source = yaml.safe_load(MODEL.read_text(encoding="utf-8"))
    cache = {}

    def run(variant, batch, prompt, tokens=128, width=1024):
        key = (variant, batch, prompt, tokens, width if variant == "sliding" else 0)
        if key not in cache:
            name = f"{variant}_b{batch}_p{prompt}_o{tokens}_w{key[-1]}"
            print(f"Simulating {name} ...", flush=True)
            row = evaluate(source, HARDWARE, output / "runs" / name, variant, tokens, batch,
                           prompt_tokens=prompt, window_size=width)
            row["max_batch"] = None
            row["run_dir"] = f"runs/{name}"
            cache[key] = row
            print(f"  prefill={row['prefill_s']:.4f}s, TPOT={row['tpot_ms']:.4f}ms, "
                  f"peak={row['peak_gib']:.4f}GiB, fits={row['fits']}", flush=True)
        return dict(cache[key])

    required = []
    for tokens in (128, 256):
        for variant in ("dense", "sliding"):
            row = run(variant, 16, 8192, tokens)
            row["max_batch"] = largest_batch(
                source, HARDWARE, output / "runs/capacity" / f"{variant}_{tokens}", variant, tokens
            )
            required.append(row)
    write_csv(output / "required.csv", required)

    prompt_rows = [run(variant, 4, prompt) for prompt in PROMPTS for variant in ("dense", "sliding")]
    write_csv(output / "prompt_sweep.csv", prompt_rows)
    window_rows = [run("dense", 4, 8192)] + [run("sliding", 4, 8192, width=width) for width in WINDOWS]
    write_csv(output / "window_sweep.csv", window_rows)

    # The whole generated sequence must fit, not just the prompt, for equivalence.
    short = [r for r in prompt_rows if r["prompt_tokens"] == 512]
    wide = [r for r in window_rows if r["attention"] == "dense" or r["window_size"] == 16384]
    for pair in (short, wide):
        for metric in ("prefill_s", "decode_s", "peak_gib"):
            assert math.isclose(pair[0][metric], pair[1][metric], rel_tol=1e-12), metric
    for row in required:
        base = output / "runs/capacity" / f"{row['attention']}_{row['output_tokens']}"
        for batch, expected in ((row["max_batch"], True), (row["max_batch"] + 1, False)):
            if batch:
                probe = json.loads((base / f"batch_{batch}/results.json").read_text())
                assert probe["fits"] == expected
    return {"required": required, "prompt_sweep": prompt_rows, "window_sweep": window_rows}


def plots(output, studies):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"dense": "#3569A8", "sliding": "#D16A26"}
    metrics = (("prefill_s", "Prefill time (s)"), ("tpot_ms", "TPOT (ms / output token)"),
               ("peak_gib", "Peak memory (GiB)"))

    def save(fig, name, note):
        fig.text(0.5, 0.015, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, 0.92))
        fig.savefig(figures / f"{name}.png", dpi=180)
        fig.savefig(figures / f"{name}.svg")
        plt.close(fig)

    required = studies["required"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    for ax, (metric, label) in zip(axes, (*metrics, ("max_batch", "Largest fitting batch"))):
        for variant, shift in (("dense", -0.18), ("sliding", 0.18)):
            rows = [r for r in required if r["attention"] == variant]
            bars = ax.bar([i + shift for i in range(2)], [r[metric] for r in rows],
                          width=0.34, label=variant.title(), color=colors[variant])
            if metric != "max_batch":
                for bar, row in zip(bars, rows):
                    if not row["fits"]:
                        bar.set_hatch("///")
            ax.bar_label(bars, fmt="%.2f" if metric != "max_batch" else "%d", fontsize=8, padding=3)
        ax.set_xticks([0, 1], ["128", "256"])
        ax.set_xlabel("Output tokens")
        ax.set_ylabel(label)
        ax.margins(y=0.2)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        if metric == "peak_gib":
            ax.axhline(80, color="#555555", linestyle=":", linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9),
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Required evaluation: Gemma 3 4B, A100 80 GiB, prompt 8192, batch 16")
    save(fig, "required_comparison", "Hatched batch-16 bars exceed capacity: timing is theoretical. Maximum batch uses a separate capacity search.")

    for name in ("prompt_sweep", "window_sweep"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        rows = studies[name]
        for ax, (metric, label) in zip(axes, metrics):
            if name == "prompt_sweep":
                ticks = PROMPTS
                for variant in ("dense", "sliding"):
                    selected = [r for r in rows if r["attention"] == variant]
                    ax.plot([r["prompt_tokens"] for r in selected], [r[metric] for r in selected],
                            marker="o", color=colors[variant], label=variant.title())
                ax.axvline(1024, color="gray", linestyle=":", linewidth=1)
                ax.set_xlabel("Prompt length (tokens)")
            else:
                ticks = WINDOWS
                selected = [r for r in rows if r["attention"] == "sliding"]
                dense = next(r for r in rows if r["attention"] == "dense")
                ax.axhline(dense[metric], linestyle="--", color=colors["dense"], label="Dense baseline")
                ax.plot([r["window_size"] for r in selected], [r[metric] for r in selected],
                        marker="o", color=colors["sliding"], label="Sliding")
                ax.set_xlabel("Local window (tokens)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks, [str(v) for v in ticks], rotation=30)
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
        title = ("Case study 1: prompt length (window 1024)" if name == "prompt_sweep"
                 else "Case study 2: window size (prompt 8192)")
        fig.suptitle(title + " | batch 4, output 128, A100 80 GiB")
        note = ("Dotted line: prompt equals window; generated output can still extend beyond the window."
                if name == "prompt_sweep" else "Window 16384 covers the entire 8320-token sequence. Model quality is not evaluated.")
        save(fig, name, note)


def report(output, studies):
    def table(rows):
        lines = ["| Attention | B | Prompt | Output | Window | Prefill (s) | TPOT (ms) | Peak (GiB) | Fits | Max B |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|"]
        for r in rows:
            lines.append(f"| {r['attention']} | {r['batch_size']} | {r['prompt_tokens']} | {r['output_tokens']} | "
                         f"{r['window_size'] or 'global'} | {r['prefill_s']:.4f} | {r['tpot_ms']:.4f} | "
                         f"{r['peak_gib']:.4f} | {'yes' if r['fits'] else 'NO'} | "
                         f"{r['max_batch'] if r['max_batch'] is not None else '-'} |")
        return "\n".join(lines)

    def select(study, variant, **values):
        return next(r for r in studies[study] if r["attention"] == variant
                    and all(r[k] == v for k, v in values.items()))

    dense = select("required", "dense", output_tokens=128)
    local = select("required", "sliding", output_tokens=128)
    long_d = select("prompt_sweep", "dense", prompt_tokens=16384)
    long_l = select("prompt_sweep", "sliding", prompt_tokens=16384)
    narrow = select("window_sweep", "sliding", window_size=256)
    wide = select("window_sweep", "sliding", window_size=16384)
    text = f"""# Project 2 results: sliding-window attention

Generated by `python reproduce.py`. These are analytical estimates, not GPU measurements.
All runs use Gemma 3 4B and the unchanged single-A100 80 GiB hardware config.
Sliding uses 29 local layers and 5 global layers (every sixth layer global).

## Required evaluation (Table 1 / Figure 1)

![Required comparison](figures/required_comparison.png)

{table(studies['required'])}

At output 128, prefill is {dense['prefill_s']/local['prefill_s']:.3f}x faster and TPOT is
{dense['tpot_ms']/local['tpot_ms']:.3f}x faster in the analytical model. Peak memory changes from
{dense['peak_gib']:.4f} to {local['peak_gib']:.4f} GiB; maximum fitting batch changes from
{dense['max_batch']} to {local['max_batch']}. Dense at batch 16 exceeds capacity: its timing is
a theoretical comparison, not deployable throughput or a measured speedup.

## Case study 1: prompt length (Table 2 / Figure 2)

Question: when does a fixed 1024-token window start helping? Batch is 4; output is 128.

![Prompt sweep](figures/prompt_sweep.png)

{table(studies['prompt_sweep'])}

At prompt 512, the entire 640-token sequence fits inside the window and dense/sliding results
match. At prompt 1024, prefill still fits, but decode extends beyond the window. At prompt
16384, dense/sliding TPOT is {long_d['tpot_ms']:.4f}/{long_l['tpot_ms']:.4f} ms
({long_d['tpot_ms']/long_l['tpot_ms']:.3f}x ratio), and peak memory is
{long_d['peak_gib']:.4f}/{long_l['peak_gib']:.4f} GiB.

Local decode attention and retained KV stop growing after window saturation. Global layers
still grow with context, and prefill must process every prompt token through projections,
FFNs and the vocabulary head. Thus total cost and peak memory do not become constant.
Tile boundaries and GEMM kernel selection can make the curves stepwise rather than smooth.

## Case study 2: window size (Table 3 / Figure 3)

Question: how does the attention budget change cost? Batch is 4, prompt 8192, output 128.

![Window sweep](figures/window_sweep.png)

{table(studies['window_sweep'])}

For window 256 versus 16384, TPOT is {narrow['tpot_ms']:.4f} versus {wide['tpot_ms']:.4f} ms,
and peak memory is {narrow['peak_gib']:.4f} versus {wide['peak_gib']:.4f} GiB. The largest window
covers all 8320 tokens and matches dense timing/memory; this equivalence is checked by the script.
Smaller windows reduce only local attention work and local KV retention. Unchanged global
layers and non-attention operations limit savings. This experiment does not measure language
model quality, so it cannot establish which window has the best accuracy/performance tradeoff.

## Reading and reproducing the data

- `required.csv`, `prompt_sweep.csv`, `window_sweep.csv`: the rows behind Tables 1-3.
- `figures/*.png`: slide-ready images; matching `.svg` files are scalable versions.
- `study_data.json`: exact values used to regenerate figures and this report.
- `manifest.json`: input/code SHA-256 hashes, environment versions and wall-clock runtime.
- `runs/<case>/model.yaml`, `results.json`, `run.log`: exact inputs, metrics and simulator logs.
- `runs/<case>/output/LLM/memory-summary/`: graph-replayed prefill/decode memory summaries.
- `runs/capacity/`: memory-only probes, including the largest fitting batch and its failing neighbor.

TPOT = total decode seconds / output tokens * 1000, per sequence. Peak is the maximum of
prefill and final-decode peaks, per GPU, including weights, KV and modeled activations.
Capacity decisions use full precision, not rounded display values. The memory-only search
uses static memory plus the largest transient allocation, equivalent to this simulator's
sequential single-GPU inference replay; it does not reprice GEMMs at every batch.

Run details are regenerated and excluded from Git; this report, CSVs, figures and manifest
are the compact shareable snapshot. No model weights, CUDA, GPU or external dataset is needed.
"""
    (output / "RESULTS.md").write_text(text, encoding="utf-8")


def manifest(elapsed):
    files = sorted(set(ROOT.glob("*.py")) | {MODEL, HARDWARE, ROOT / "requirements.txt"})
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "PyYAML", "graphviz", "matplotlib")},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "wall_clock_s": elapsed,
        "command": "python reproduce.py",
        "experiments": {"prompts": PROMPTS, "windows": WINDOWS, "case_study_batch": 4,
                        "case_study_output": 128, "required_batch": 16, "required_outputs": [128, 256]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/project2")
    parser.add_argument("--plots-only", action="store_true", help="Regenerate figures/report from existing study_data.json")
    parser.add_argument("--verify-only", action="store_true", help="Run quick correctness checks, then exit")
    args = parser.parse_args()
    if args.verify_only:
        verify()
        return
    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "study_data.json"
    if args.plots_only:
        studies = json.loads(data_path.read_text(encoding="utf-8"))
    else:
        verify()
        studies = run_experiments(output)
        data_path.write_text(json.dumps(studies, indent=2) + "\n", encoding="utf-8")
    plots(output, studies)
    report(output, studies)
    if not args.plots_only:
        (output / "manifest.json").write_text(json.dumps(manifest(time.perf_counter()-started), indent=2) + "\n", encoding="utf-8")
    elif (output / "manifest.json").exists():
        metadata = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        metadata["plot_regeneration"] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "command": "python reproduce.py --plots-only",
        }
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {output / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
