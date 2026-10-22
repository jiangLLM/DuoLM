# Third-party notices

Original DuoLM code is licensed under the MIT License in [LICENSE](LICENSE).
External libraries, model weights, and datasets retain their own licenses.

## SGLang

The source-anchor and replacement snippets in
`src/duolm/nextbook_sglang_overlay.py` adapt SGLang source. SGLang is
Copyright 2023 SGLang Team, licensed under the Apache License, Version 2.0.
The license is reproduced in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
See the [upstream project](https://github.com/sgl-project/sglang).

The DuoLM overlay modifies a separately installed SGLang distribution to
provide item-vector lookup, scalar recommendation scores, isolated candidate
branches, and the associated CUDA Graph execution path. It checks the exact
SGLang version and source hashes before writing a separate overlay directory.
These modifications are not part of an unmodified upstream installation.

## Pretrained models

Training uses [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
and [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
Both are external downloads; their weights and tokenizer assets are not
included in this repository. Consult their model cards and licenses.

## Data

Experiments use [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).
This repository includes preparation code and synthetic test fixtures.
It does not redistribute Amazon review records, user histories, item catalogs,
or derived teacher predictions. Consult the dataset provider for access,
terms, and citation instructions.
