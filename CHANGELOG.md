<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Changelog

## 0.2.3

- Documentation: `engine_benchmark` forces the output length through the
  `force_generation_tokens` scenario option, which the harness must set.

## 0.2.2

- `scripts/vlm_benchmark.py`: remove the early return after the first token.
  The decode loop already does not run when the generator is done, and the
  generated-token check after the loop fails the run in the same way. Behavior
  is unchanged.

## 0.2.1

- `scripts/vlm_benchmark.py`: the measured iterations no longer read back or
  decode generated tokens. With `-v` the generated text is printed by the first
  warmup iteration only, as `model_benchmark` does; with `-w 0` nothing is
  printed. `run_single_benchmark` now returns only the `BenchmarkResult`.
- Reported numbers are unchanged: the removed readback ran outside the timed
  intervals, but it added host work between decode steps when `-v` was set.

## 0.2.0

- `scripts/vlm_benchmark.py`: every run, warmup included, now sets
  `min_length` to prompt length + `max_tokens`, so end-of-sequence can no longer
  end a run early.
- The script fails before the first run when prompt length + `max_tokens`
  exceeds `max_length`, and fails any run whose generated-token count differs
  from `max_tokens`.
- Effect on reported numbers: runs that previously stopped early are no longer
  possible, so `avg_tokens_per_run` always equals `max_tokens` and the decode
  statistics always have `max_tokens - 1` samples per run.

## 0.1.0

- Initial import of `scripts/vlm_benchmark.py`.
- Document the benchmark families, the alignment rules and the per-tool status.
