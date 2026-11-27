<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Alignment of the benchmark tools

Status of each tool against the rules in the README. Paths for OGA tools are
relative to the onnxruntime-genai repository.

## Token accounting

A run of N generated tokens is one prefill forward pass plus N - 1 decode
forward passes. The first token is sampled from the prefill logits, so no
forward pass runs for it. The last token is sampled by the final decode step,
and no forward pass runs for it either.

| Tool | Where the first token is timed | Decode samples for N = 128 | Evidence |
|---|---|---|---|
| `model_benchmark` | Separate sampling timer | 127 | `benchmark/c/main.cpp`: the decode sample vector is reserved for `N - 1` per iteration; the loop runs until `TokenCount()` reaches prompt length + N |
| `benchmark_multimodal.py` | Time to first token | 127 | The decode loop starts at `i = 1` and stops at `i < generation_length` |
| `scripts/vlm_benchmark.py` | Time to first token | 127 | `total_tokens_generated` is set to 1 after the first token; only the decode loop appends to `token_gen_times_ms`; the loop stops when `total_tokens_generated == max_tokens` |
| `engine_benchmark` | First token event of a request | 127 | The first token event sets the first-token time; later token events append inter-token latency |

`avg_tokens_per_run` therefore counts the first token and equals N, while the
decode statistics have N - 1 samples. These are two different quantities.

## Throughput aggregation

| Tool | Statistic used for tokens/s |
|---|---|
| `model_benchmark` | `1 / mean` over all pooled decode samples |
| `benchmark_multimodal.py` | `1 / mean` over all pooled decode samples |
| `scripts/vlm_benchmark.py` | `1000 / mean` over all pooled decode samples |
| `engine_benchmark` | The tool reports the p50 of the inter-token latency; the CI consumer takes the per-request p50. The mean is only available with a local change to the tool, so this tool does not match the others yet. |

## Early stop on end-of-sequence

| Tool | Behavior |
|---|---|
| `model_benchmark` | Sets `min_length` to prompt length + N, so end-of-sequence is suppressed |
| `benchmark_multimodal.py` | Sets `min_length` to `max_length`, so end-of-sequence is suppressed |
| `scripts/vlm_benchmark.py` | Sets `min_length` to prompt length + `max_tokens` (warmup included), fails before the run if that exceeds `max_length`, and fails a run that generated a different number of tokens than `max_tokens`. |
| `engine_benchmark` | Fails the whole case when a turn finishes early, unless `ENGINE_BENCH_MIN_TOKENS=1`, which sets a minimum generated-token count on every turn |

## Time to first token

The start of the TTFT interval differs between tools. TTFT values are not
comparable across tools until these differences are removed or documented.

| Tool | Interval starts | Includes |
|---|---|---|
| `model_benchmark` | Before prompt append | Prefill only. First-token sampling is timed separately. |
| `benchmark_multimodal.py` | Before `Generator` creation | `Generator` creation, input setup, prefill, first-token sampling. Input preprocessing is timed separately. |
| `scripts/vlm_benchmark.py` | Before `set_inputs` | Input setup, prefill, first-token sampling. Preprocessing is timed separately. |
| `engine_benchmark` | Before request creation | Request creation, admission, prefill, first-token event |

## Engine API with images

The OGA Engine vision example is a qualification reproduction. It fixes the
prompt template, the generated length (16 to 32 tokens) and the statistic
(median decode interval), and its TTFT starts before the turn is submitted. It
is not suitable as a benchmark as is. A vision-capable Engine benchmark has to
be maintained in this repository or added to `engine_benchmark`.

## Open items

- `min_length` only masks the end-of-sequence tokens listed in the model
  config. Every model used for measurement must list all of its stop tokens
  there; the generated-token check in `scripts/vlm_benchmark.py` catches a
  model that does not.
- `engine_benchmark`: report the mean inter-token latency and have the
  consumer read it; `avg_tokens_per_run` should be a measured value.
- Define one TTFT interval and report the others as separate fields.
- Add a vision-capable Engine benchmark.
