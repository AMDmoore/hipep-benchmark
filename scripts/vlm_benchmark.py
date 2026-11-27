# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
VLM Benchmark Script for onnxruntime-genai

Measures:
- TTFT (Time To First Token): Time from inputs ready to first token generated
- TPS (Tokens Per Second): Token generation throughput
- Preprocessing Time: Time spent in multimodal processor (image preprocessing)
- Image Token Count: Number of tokens generated from the image by the vision encoder

Supports a text-only mode (``--text_only``) that drives the same multimodal
model without any image input, so the text pipeline can be measured on its own.
"""

import argparse
import json
import os
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import onnxruntime_genai as og


DEFAULT_IMAGE_PROMPT = "Describe this image in detail."
DEFAULT_TEXT_ONLY_PROMPT = "Explain the theory of relativity in detail."


def log_gpu_device_visibility() -> None:
    """Log the Linux GPU device nodes visible to this process."""
    interesting_env = (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_OVERRIDE_GFX_VERSION",
    )
    env_summary = ", ".join(
        f"{name}={os.environ.get(name, '<unset>')}" for name in interesting_env
    )
    print(f"GPU visibility env: {env_summary}", flush=True)

    if os.name != "posix":
        return

    kfd = Path("/dev/kfd")
    dri = Path("/dev/dri")
    dri_nodes = sorted(path.name for path in dri.glob("*")) if dri.is_dir() else []
    print(f"/dev/kfd visible: {kfd.exists()}", flush=True)
    print(
        "/dev/dri nodes: " + (", ".join(dri_nodes) if dri_nodes else "<none>"),
        flush=True,
    )
    _probe_hip_devices()
    _log_loaded_runtime_libs()


def _probe_hip_devices() -> None:
    """Query the HIP runtime directly so we can tell whether the process itself
    can enumerate the GPU independent of the onnxruntime-genai EP.
    """
    import ctypes

    hip = None
    for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            hip = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if hip is None:
        print("[hip-probe] could not load libamdhip64.so", flush=True)
        return
    try:
        count = ctypes.c_int(0)
        rc = hip.hipGetDeviceCount(ctypes.byref(count))
        print(f"[hip-probe] hipGetDeviceCount rc={rc} count={count.value}", flush=True)
    except Exception as exc:
        print(f"[hip-probe] hipGetDeviceCount failed: {exc}", flush=True)


def _log_loaded_runtime_libs() -> None:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return
    keys = (
        "libamdhip64",
        "libhsa-runtime64",
        "libhipgpu",
        "libamdgpu-ep",
        "libonnxruntime-genai",
        "libonnxruntime.so",
    )
    seen: set[str] = set()
    try:
        for line in maps.read_text(errors="replace").splitlines():
            path = line.rsplit(" ", 1)[-1]
            if not path.startswith("/"):
                continue
            if any(key in path for key in keys) and path not in seen:
                seen.add(path)
                print(f"[loaded-lib] {path}", flush=True)
    except OSError:
        pass


def get_peak_memory_bytes() -> int:
    """Get the process-lifetime peak working set in bytes."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    if os.name == "posix":
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    raise OSError(f"Peak working set is not supported on platform {os.name!r}")


@dataclass
class BenchmarkResult:
    """Stores benchmark measurements for a single run."""

    preprocessing_time_ms: float = 0.0
    ttft_ms: float = 0.0
    token_gen_times_ms: list = field(default_factory=list)
    total_tokens_generated: int = 0
    prompt_tokens: int = 0
    image_tokens: int = 0
    text_tokens: int = 0


@dataclass
class BenchmarkStats:
    """Aggregated statistics across multiple runs."""

    avg: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    n: int = 0


def compute_stats(measurements: list) -> BenchmarkStats:
    """Compute statistics from a list of measurements."""
    if not measurements:
        return BenchmarkStats()

    sorted_m = sorted(measurements)
    n = len(sorted_m)

    stats = BenchmarkStats(
        avg=statistics.mean(sorted_m),
        std=statistics.stdev(sorted_m) if n > 1 else 0.0,
        p50=sorted_m[int(n * 0.5)] if n > 0 else 0.0,
        p90=sorted_m[int(n * 0.9)] if n > 0 else 0.0,
        p99=sorted_m[int(n * 0.99)] if n > 0 else 0.0,
        min_val=min(sorted_m),
        max_val=max(sorted_m),
        n=n,
    )
    return stats


def resize_image(image_path: str, size: tuple) -> str:
    """Resize image to target size and save to temp file."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(tmp.name, format="JPEG")
    return tmp.name


def get_image_size(image_path: str) -> tuple:
    """Get original image dimensions."""
    with Image.open(image_path) as img:
        return img.size


def build_prompt_phi4(image_paths: list, text: str) -> str:
    """Build prompt for phi-4 multimodal model."""
    prompt = "<|user|>\n"
    for i in range(len(image_paths)):
        prompt += f"<|image_{i + 1}|>\n"
    prompt += f"{text}<|end|>\n<|assistant|>\n"
    return prompt


def build_prompt_chat_template(
    model_path: str, tokenizer: og.Tokenizer, image_paths: list, text: str
) -> str:
    """Build prompt using chat_template from the model directory.

    Looks for the template in two locations (in order):
      1. "chat_template" field inside tokenizer_config.json (inline)
      2. chat_template.jinja file (common after ONNX/HuggingFace export)
    """
    model_dir = Path(model_path)
    tok_cfg_path = model_dir / "tokenizer_config.json"
    jinja_path = model_dir / "chat_template.jinja"

    template_str = None
    bos = None

    # 1. Try inline chat_template in tokenizer_config.json
    if tok_cfg_path.exists():
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        template_str = tok_cfg.get("chat_template")
        bos = tok_cfg.get("bos_token")

    # 2. Fall back to chat_template.jinja file
    if not template_str and jinja_path.exists():
        with open(jinja_path, "r", encoding="utf-8") as f:
            template_str = f.read()
        print(f"  Loaded chat template from: {jinja_path}")

    if not template_str:
        raise RuntimeError(
            f"No chat template found at {model_path}. "
            f"Looked in tokenizer_config.json and chat_template.jinja. "
            f"Please add a prompt builder for this model type."
        )

    if not bos:
        template_str = template_str.replace("{{ bos_token }}", "")

    content_list = [{"type": "image"} for _ in image_paths]
    content_list.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content_list}]

    message_json = json.dumps(messages)
    prompt = tokenizer.apply_chat_template(
        message_json, template_str=template_str, add_generation_prompt=True
    )
    return prompt


def count_text_tokens(tokenizer: og.Tokenizer, prompt: str) -> int:
    """Count text-only tokens in the prompt (without image tokens).

    Encodes the raw prompt string to get the text token count.
    Image tokens are added by the multimodal processor on top of this.
    """
    try:
        text_ids = tokenizer.encode(prompt)
        return len(text_ids)
    except Exception:
        return 0


def process_inputs(
    processor: og.MultiModalProcessor, prompt: str, images: Optional[og.Images]
):
    """Encode *prompt* with optional *images* into model inputs.

    Text-only runs pass the prompt on its own rather than ``images=None`` so the
    processor never sees an image argument it may not accept.
    """
    if images is None:
        return processor(prompt)
    return processor(prompt, images=images)


def make_generator_params(
    model: og.Model, inputs, max_tokens: int, max_length: int
) -> og.GeneratorParams:
    """Build generator parameters that force exactly *max_tokens* generated tokens.

    ``min_length`` is a floor on the total sequence length, prompt included, and
    only masks the end-of-sequence tokens listed in the model config. Setting it
    to prompt length + max_tokens keeps a run from ending early; the decode loop
    provides the upper bound.
    """
    if "input_ids" not in inputs.keys():
        raise RuntimeError("Processor output has no input_ids; cannot derive min_length")
    input_ids = inputs["input_ids"]
    shape = input_ids.shape()
    prompt_length = shape[1] if len(shape) > 1 else shape[0]
    min_length = prompt_length + max_tokens
    if min_length > max_length:
        raise ValueError(
            f"prompt length {prompt_length} + max_tokens {max_tokens} exceeds "
            f"max_length {max_length}"
        )
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length, min_length=min_length)
    return params


def require_full_generation(result: "BenchmarkResult", max_tokens: int) -> None:
    """Fail the run when it produced fewer or more than *max_tokens* tokens."""
    if result.total_tokens_generated != max_tokens:
        raise RuntimeError(
            f"Run generated {result.total_tokens_generated} tokens, expected {max_tokens}"
        )


def run_single_benchmark(
    model: og.Model,
    processor: og.MultiModalProcessor,
    prompt: str,
    images: Optional[og.Images],
    max_tokens: int,
    max_length: int,
    tokenizer: Optional[og.Tokenizer] = None,
) -> "BenchmarkResult":
    """Run a single benchmark iteration and collect timing metrics.

    Nothing is read back from the generator besides the timing: the generated
    text is printed by the warmup run, which uses the same prompt and parameters.
    """
    result = BenchmarkResult()

    # Measure preprocessing time (multimodal processor call)
    preprocess_start = time.perf_counter()
    inputs = process_inputs(processor, prompt, images)
    preprocess_end = time.perf_counter()
    result.preprocessing_time_ms = (preprocess_end - preprocess_start) * 1000

    # Get prompt token count from inputs (includes image tokens post-ViT)
    if "input_ids" in inputs.keys():
        input_ids = inputs["input_ids"]
        result.prompt_tokens = (
            input_ids.shape()[1] if len(input_ids.shape()) > 1 else input_ids.shape()[0]
        )

    # Compute image token count = total prompt tokens - text-only tokens
    if tokenizer is not None:
        result.text_tokens = count_text_tokens(tokenizer, prompt)
        result.image_tokens = (
            0 if images is None else max(0, result.prompt_tokens - result.text_tokens)
        )

    # Setup generator
    params = make_generator_params(model, inputs, max_tokens, max_length)

    generator = og.Generator(model, params)

    # Measure TTFT: includes set_inputs (prefill) + first token generation
    ttft_start = time.perf_counter()
    generator.set_inputs(inputs)
    generator.generate_next_token()
    ttft_end = time.perf_counter()
    result.ttft_ms = (ttft_end - ttft_start) * 1000
    result.total_tokens_generated = 1

    # Measure subsequent token generation times
    while not generator.is_done() and result.total_tokens_generated < max_tokens:
        token_start = time.perf_counter()
        generator.generate_next_token()
        token_end = time.perf_counter()
        result.token_gen_times_ms.append((token_end - token_start) * 1000)
        result.total_tokens_generated += 1

    del generator
    require_full_generation(result, max_tokens)
    return result


def run_warmup(
    model: og.Model,
    processor: og.MultiModalProcessor,
    prompt: str,
    images: Optional[og.Images],
    max_tokens: int,
    max_length: int,
    num_iterations: int,
    verbose: bool = False,
):
    """Run warmup iterations."""
    if verbose:
        print(f"Running {num_iterations} warmup iteration(s)...")

    for i in range(num_iterations):
        inputs = process_inputs(processor, prompt, images)
        params = make_generator_params(model, inputs, max_tokens, max_length)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        tokens_generated = 0
        stream = processor.create_stream()

        while not generator.is_done() and tokens_generated < max_tokens:
            generator.generate_next_token()
            tokens_generated += 1

            if verbose and i == 0:
                new_token = generator.get_next_tokens()[0]
                print(stream.decode(new_token), end="", flush=True)

        if verbose and i == 0:
            print("\n")

        del generator


def print_benchmark_results(
    results: list,
    image_size: Optional[tuple],
    model_type: str,
    peak_memory_bytes: int,
):
    """Print formatted benchmark results."""
    print("\n" + "=" * 60)
    print("VLM BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model Type: {model_type}")
    if image_size is None:
        print("Input Mode: text-only (no image)")
    else:
        print(f"Image Size: {image_size[0]}x{image_size[1]}")
    print(f"Number of runs: {len(results)}")
    print("-" * 60)

    # Token breakdown (from first run)
    if results and results[0].prompt_tokens > 0:
        r0 = results[0]
        print(f"\nToken Breakdown:")
        print(f"  Total Prompt Tokens: {r0.prompt_tokens}")
        print(f"  Text Tokens:         {r0.text_tokens}")
        print(f"  Image Tokens:        {r0.image_tokens}")

    # Preprocessing Time
    preprocess_times = [r.preprocessing_time_ms for r in results]
    preprocess_stats = compute_stats(preprocess_times)
    preprocess_label = "text processing" if image_size is None else "image processing"
    print(f"\nPreprocessing Time ({preprocess_label}):")
    print(f"  Average:  {preprocess_stats.avg:>10.2f} ms")
    print(f"  Std Dev:  {preprocess_stats.std:>10.2f} ms")
    print(f"  P50:      {preprocess_stats.p50:>10.2f} ms")
    print(f"  P90:      {preprocess_stats.p90:>10.2f} ms")
    print(f"  P99:      {preprocess_stats.p99:>10.2f} ms")
    print(f"  Min:      {preprocess_stats.min_val:>10.2f} ms")
    print(f"  Max:      {preprocess_stats.max_val:>10.2f} ms")

    # TTFT
    ttft_times = [r.ttft_ms for r in results]
    ttft_stats = compute_stats(ttft_times)
    print(f"\nTime To First Token (TTFT):")
    print(f"  Average:  {ttft_stats.avg:>10.2f} ms")
    print(f"  Std Dev:  {ttft_stats.std:>10.2f} ms")
    print(f"  P50:      {ttft_stats.p50:>10.2f} ms")
    print(f"  P90:      {ttft_stats.p90:>10.2f} ms")
    print(f"  P99:      {ttft_stats.p99:>10.2f} ms")
    print(f"  Min:      {ttft_stats.min_val:>10.2f} ms")
    print(f"  Max:      {ttft_stats.max_val:>10.2f} ms")

    # Prefill throughput (tokens/s)
    if results and results[0].prompt_tokens > 0:
        prefill_tps_list = []
        for r in results:
            if r.ttft_ms > 0:
                prefill_tps_list.append(r.prompt_tokens / (r.ttft_ms / 1000.0))
        if prefill_tps_list:
            prefill_tps_stats = compute_stats(prefill_tps_list)
            print(f"\nPrefill Throughput:")
            print(f"  Average:  {prefill_tps_stats.avg:>10.2f} tokens/sec")
            print(f"  P50:      {prefill_tps_stats.p50:>10.2f} tokens/sec")

    # Token Generation (TPS)
    all_token_times = []
    for r in results:
        all_token_times.extend(r.token_gen_times_ms)

    if all_token_times:
        token_stats = compute_stats(all_token_times)
        avg_tps = 1000.0 / token_stats.avg if token_stats.avg > 0 else 0
        print(f"\nToken Generation (excluding first token):")
        print(f"  Average:  {token_stats.avg:>10.2f} ms/token")
        print(f"  TPS:      {avg_tps:>10.2f} tokens/sec")
        print(f"  Std Dev:  {token_stats.std:>10.2f} ms")
        print(
            f"  P50:      {token_stats.p50:>10.2f} ms/token ({1000.0/token_stats.p50:.2f} TPS)"
            if token_stats.p50 > 0
            else "  P50:      N/A"
        )
        print(
            f"  P90:      {token_stats.p90:>10.2f} ms/token ({1000.0/token_stats.p90:.2f} TPS)"
            if token_stats.p90 > 0
            else "  P90:      N/A"
        )
        print(f"  Min:      {token_stats.min_val:>10.2f} ms/token")
        print(f"  Max:      {token_stats.max_val:>10.2f} ms/token")

    # Total tokens
    total_tokens = sum(r.total_tokens_generated for r in results)
    avg_tokens = total_tokens / len(results) if results else 0
    print(f"\nTokens Generated:")
    print(f"  Average per run: {avg_tokens:.1f}")
    print(f"  Total:           {total_tokens}")

    # E2E metrics
    e2e_times = [
        r.preprocessing_time_ms + r.ttft_ms + sum(r.token_gen_times_ms) for r in results
    ]
    e2e_stats = compute_stats(e2e_times)
    print(f"\nEnd-to-End Time (Preprocessing + TTFT + Token Gen):")
    print(f"  Average:  {e2e_stats.avg:>10.2f} ms")
    print(f"  P50:      {e2e_stats.p50:>10.2f} ms")

    # Peak memory
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    peak_memory_gb = peak_memory_bytes / (1024 * 1024 * 1024)
    print(f"\nPeak Memory Usage:")
    print(f"  {peak_memory_mb:>10.2f} MB ({peak_memory_gb:.2f} GB)")

    print("=" * 60 + "\n")


def export_results_json(
    results: list, output_path: str, config: dict, peak_memory_bytes: int
):
    """Export benchmark results to JSON file."""
    preprocess_times = [r.preprocessing_time_ms for r in results]
    ttft_times = [r.ttft_ms for r in results]
    all_token_times = []
    for r in results:
        all_token_times.extend(r.token_gen_times_ms)

    preprocess_stats = compute_stats(preprocess_times)
    ttft_stats = compute_stats(ttft_times)
    token_stats = (
        compute_stats(all_token_times) if all_token_times else BenchmarkStats()
    )

    # Prefill throughput
    prefill_tps_list = []
    for r in results:
        if r.ttft_ms > 0 and r.prompt_tokens > 0:
            prefill_tps_list.append(r.prompt_tokens / (r.ttft_ms / 1000.0))
    prefill_tps_stats = (
        compute_stats(prefill_tps_list) if prefill_tps_list else BenchmarkStats()
    )

    # Token counts from first run
    first_run = results[0] if results else BenchmarkResult()

    export_data = {
        "config": config,
        "summary": {
            "token_breakdown": {
                "prompt_tokens": first_run.prompt_tokens,
                "text_tokens": first_run.text_tokens,
                "image_tokens": first_run.image_tokens,
            },
            "preprocessing_ms": {
                "avg": preprocess_stats.avg,
                "std": preprocess_stats.std,
                "p50": preprocess_stats.p50,
                "p90": preprocess_stats.p90,
                "p99": preprocess_stats.p99,
                "min": preprocess_stats.min_val,
                "max": preprocess_stats.max_val,
            },
            "ttft_ms": {
                "avg": ttft_stats.avg,
                "std": ttft_stats.std,
                "p50": ttft_stats.p50,
                "p90": ttft_stats.p90,
                "p99": ttft_stats.p99,
                "min": ttft_stats.min_val,
                "max": ttft_stats.max_val,
            },
            "prefill_throughput_tps": {
                "avg": prefill_tps_stats.avg,
                "p50": prefill_tps_stats.p50,
            },
            "token_generation_ms": {
                "avg": token_stats.avg,
                "std": token_stats.std,
                "p50": token_stats.p50,
                "p90": token_stats.p90,
                "tps_avg": 1000.0 / token_stats.avg if token_stats.avg > 0 else 0,
            },
            "total_runs": len(results),
            "avg_tokens_per_run": sum(r.total_tokens_generated for r in results)
            / len(results)
            if results
            else 0,
            "peak_memory_bytes": peak_memory_bytes,
            "peak_memory_mb": peak_memory_bytes / (1024 * 1024),
        },
        "raw_measurements": {
            "preprocessing_times_ms": preprocess_times,
            "ttft_times_ms": ttft_times,
            "token_gen_times_ms": all_token_times,
        },
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"Results exported to: {output_path}")


def run_benchmark(args: argparse.Namespace):
    """Main benchmark function."""
    print("Loading model...")
    log_gpu_device_visibility()
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            print(f"Setting execution provider to {args.execution_provider}...")
            config.append_provider(args.execution_provider)

    model = og.Model(config)
    model_type = model.type
    print(f"Model loaded: {model_type}")
    print(f"Device: {model.device_type}")

    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)

    if args.text_only:
        print("Text-only mode: skipping image input")
        image_paths = []
        images = None
        target_size = None
    else:
        # Validate image path
        if not args.image_path or not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image file not found: {args.image_path}")

        # Get original image size
        original_size = get_image_size(args.image_path)
        print(f"Original image size: {original_size[0]}x{original_size[1]}")

        # Determine target image size
        if args.image_size:
            target_size = tuple(args.image_size)
        else:
            target_size = original_size
        print(f"Benchmark image size: {target_size[0]}x{target_size[1]}")

        # Resize image if needed
        if target_size != original_size:
            print(f"Resizing image to {target_size[0]}x{target_size[1]}...")
            resized_path = resize_image(args.image_path, target_size)
            images = og.Images.open(resized_path)
        else:
            images = og.Images.open(args.image_path)
        image_paths = [args.image_path]

    # Build prompt based on model type
    default_prompt = (
        DEFAULT_TEXT_ONLY_PROMPT if args.text_only else DEFAULT_IMAGE_PROMPT
    )
    prompt_text = args.prompt if args.prompt else default_prompt

    if model_type in ("phi3v", "phi4mm"):
        prompt = build_prompt_phi4(image_paths, prompt_text)
    else:
        # All other models: use chat_template from tokenizer_config.json
        # Works for any model that ships a chat template.
        print(
            f"Model type: '{model_type}' — using chat template from tokenizer_config.json"
        )
        prompt = build_prompt_chat_template(
            args.model_path, tokenizer, image_paths, prompt_text
        )

    if args.verbose:
        print(f"\nPrompt: {prompt[:200]}...")

    # Run warmup
    run_warmup(
        model=model,
        processor=processor,
        prompt=prompt,
        images=images,
        max_tokens=args.max_tokens,
        max_length=args.max_length,
        num_iterations=args.warmup_iterations,
        verbose=args.verbose,
    )

    # Run benchmark iterations
    print(f"\nRunning {args.num_iterations} benchmark iteration(s)...")
    results = []

    for i in range(args.num_iterations):
        if args.verbose:
            print(f"  Iteration {i + 1}/{args.num_iterations}...", end=" ")

        result = run_single_benchmark(
            model=model,
            processor=processor,
            prompt=prompt,
            images=images,
            max_tokens=args.max_tokens,
            max_length=args.max_length,
            tokenizer=tokenizer,
        )
        results.append(result)

        # Print token breakdown on first iteration
        if i == 0:
            print(
                f"  Token breakdown: total={result.prompt_tokens} "
                f"(text={result.text_tokens}, image={result.image_tokens})"
            )

        if args.verbose:
            print(
                f"  Iteration {i + 1}: "
                f"Preprocess: {result.preprocessing_time_ms:.1f}ms, "
                f"TTFT: {result.ttft_ms:.1f}ms, "
                f"Tokens: {result.total_tokens_generated}"
            )
    # Get peak memory usage
    peak_memory_bytes = get_peak_memory_bytes()

    # Print results
    print_benchmark_results(results, target_size, model_type, peak_memory_bytes)

    # Export to JSON if requested
    if args.output_json:
        export_config = {
            "model_path": args.model_path,
            "model_type": model_type,
            "device_type": model.device_type,
            "execution_provider": args.execution_provider,
            "text_only": bool(args.text_only),
            "image_path": "" if args.text_only else args.image_path,
            "image_size": [] if target_size is None else list(target_size),
            "prompt": prompt_text,
            "max_tokens": args.max_tokens,
            "max_length": args.max_length,
            "num_iterations": args.num_iterations,
            "warmup_iterations": args.warmup_iterations,
        }
        export_results_json(results, args.output_json, export_config, peak_memory_bytes)


def main():
    parser = argparse.ArgumentParser(
        description="VLM Benchmark Script - Measure TTFT, TPS, and preprocessing time"
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to the folder containing the ONNX model",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        help="Path to the input image. Required unless --text_only is set.",
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Benchmark the model with a text-only prompt and no image input",
    )
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        default="follow_config",
        choices=["cpu", "dml", "follow_config"],
        help="Execution provider (default: follow_config)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help=(
            "Prompt text for the VLM. Defaults to "
            f"'{DEFAULT_IMAGE_PROMPT}' with an image and "
            f"'{DEFAULT_TEXT_ONLY_PROMPT}' in text-only mode."
        ),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Target image size (width height). If not specified, uses original size.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help="Max context length for KV cache (default: 16384)",
    )
    parser.add_argument(
        "-n",
        "--num_iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)",
    )
    parser.add_argument(
        "-w",
        "--warmup_iterations",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )
    parser.add_argument(
        "-o", "--output_json", type=str, help="Path to export results as JSON"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()
    if args.text_only and (args.image_path or args.image_size):
        parser.error("--text_only cannot be combined with --image_path/--image_size")
    if not args.text_only and not args.image_path:
        parser.error("--image_path is required unless --text_only is set")
    run_benchmark(args)


if __name__ == "__main__":
    main()
