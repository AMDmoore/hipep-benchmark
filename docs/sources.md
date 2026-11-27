<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Sources of maintained scripts

## `scripts/vlm_benchmark.py`

- Origin: the VLM benchmark script used by the CI GenAI flow. The file carries
  the onnxruntime-genai copyright and MIT license header and is kept as is.
- Initial import: copy of the CI script at CI repository revision `3364d40e`
  (`dev`). Line endings were normalized to LF.
- Local changes against that revision:
  - One code comment that listed model-family names now reads "Works for any
    model that ships a chat template."
- Behavior is otherwise unchanged. Later behavior changes are listed in
  `CHANGELOG.md`.

## OGA tools used as references (not copied)

- `benchmark/c` (`model_benchmark`)
- `benchmark/engine` (`engine_benchmark`)
- `benchmark/python/benchmark_multimodal.py`

Record the OGA branch and revision next to every published result.
