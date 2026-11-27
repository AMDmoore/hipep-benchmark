<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# hipep-benchmark

Measurement protocol and maintained scripts for benchmarking models on the
hip-ep execution provider through onnxruntime-genai (OGA).

Throughput numbers from different tools are only comparable when the tools count
the same things. This repository records which tool measures which workload,
how the tools are aligned, and which scripts are maintained here instead of
coming from OGA.

## Benchmark families

| Workload | Tool | Source |
|---|---|---|
| Text LLM, `Generator` API | `model_benchmark` | OGA (`benchmark/c`) |
| Text LLM, paged Engine API | `engine_benchmark` | OGA (`benchmark/engine`) |
| VLM (image or text-only), `Generator` API, CI | `scripts/vlm_benchmark.py` | Maintained here |
| VLM, `Generator` API, reference | `benchmark_multimodal.py` | OGA (`benchmark/python`) |
| VLM, paged Engine API | not available yet | See [docs/alignment.md](docs/alignment.md) |

- OGA tools are not copied here. Record the OGA revision used in
  [docs/sources.md](docs/sources.md) and in the result file.
- Scripts under `scripts/` are maintained here. Each one has an entry in
  [docs/sources.md](docs/sources.md) with its origin and local changes.

## Alignment rules

All tools must agree on these points; the details and the evidence for each
tool are in [docs/alignment.md](docs/alignment.md).

1. A run generates exactly N tokens (N = 128 by default). The first token is
   produced right after prefill and belongs to TTFT.
2. Decode statistics cover the remaining N - 1 tokens (127 samples for N = 128).
3. Throughput is `1000 / mean(decode interval in ms)` over all decode samples
   of all measured runs. Percentiles are reported next to it, never instead of it.
4. A run must not stop early on end-of-sequence. Tools either force the minimum
   length or fail the run when it ends early.
5. No performance or debug tracing is enabled while measuring.

## Running `scripts/vlm_benchmark.py`

```
python scripts/vlm_benchmark.py -m <model_dir> -i <image> \
    --max_tokens 128 -n 5 -w 1 -o <results.json>
python scripts/vlm_benchmark.py -m <model_dir> --text_only \
    --max_tokens 128 -n 5 -w 1 -o <results.json>
```

The JSON result contains `ttft_ms`, `token_generation_ms` (including `tps_avg`),
`total_runs` and `avg_tokens_per_run`.

## Versioning

- `CHANGELOG.md` records every change to a script or to a rule in this protocol.
- CI and local runs should reference a tag of this repository, so that a result
  can be traced to the exact script and protocol it was measured with.
- When a script changes how it counts tokens or computes statistics, bump the
  minor version and state the effect on reported numbers in the changelog.

## Layout

```
README.md          overview and alignment rules
CHANGELOG.md       version history
docs/alignment.md  per-tool evidence for the alignment rules, open differences
docs/sources.md    origin of each maintained script
scripts/           scripts maintained here
```
