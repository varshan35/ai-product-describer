# AI Product Describer

A decoder-only transformer language model built and trained from scratch in PyTorch to generate e-commerce product descriptions.

## Overview

The model implements a complete autoregressive transformer pipeline:
- **Tokenizer**: Byte-Pair Encoding (BPE) subword tokenizer with special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
- **Architecture**: Learned token and positional embeddings, multi-head causal self-attention with lower-triangular masking, and pre-norm residual transformer blocks.
- **Decoding**: Autoregressive generation with temperature scaling and top-k sampling.
- **Persistence**: Model weights and tokenizer state are saved to disk (`mini_gpt.pt`) for fast inference.

## Installation

```bash
git clone https://github.com/varshan35/ai-product-describer.git
cd ai-product-describer
pip install -r requirements.txt
```

## Usage

Generate descriptions from the command line:

```bash
# Standard generation
python -m app.cli generate "vintage leather"

# Adjust length and sampling temperature
python -m app.cli generate "wireless headphones" -n 40 -t 0.7 --top-k 5

# Reproducible output using a seed
python -m app.cli generate "ceramic mug" --seed 42
```

To view all available options:
```bash
python -m app.cli generate --help
```

## Testing

Run the automated test suite:

```bash
pytest
```

All 30 unit and integration tests cover tokenization roundtrips, attention causality, training loss reduction, checkpoint caching, and CLI functionality.
