# AI Product Describer

This AI Product Describer is a CLI tool designed for small business owners in the retail domain. It generates creative product descriptions for inventory by training a custom GPT on examples of existing product texts.

## What you are building

**Goal.** You will have built a command-line interface (CLI) tool that generates creative product descriptions for small business owners by training a custom GPT model on example texts.

**Done when.** The project is done when you can apply the byte-pair tokenizer and embeddings to prepare raw text, implement causal self-attention, assemble the transformer block, and train your custom GPT model from scratch to generate coherent product descriptions via a functional CLI.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Collect a small corpus of product descriptions for a specific category.**  
      Hint: Collect a focused dataset of product descriptions for a single category.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings for product text.**  
      Hint: Implement the byte-pair encoding (BPE) algorithm and create embedding layers.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Construct the self-attention mechanism with a causal mask and stack it into a Transformer block.
- [ ] **4. Train the GPT model using the collected product descriptions.**  
      Hint: Train your GPT model using a language modeling objective on your collected corpus.
- [ ] **5. Implement a text generation feature that creates new product descriptions from a prompt.**  
      Hint: Implement a sampling strategy (e.g., top-k or nucleus sampling) to generate new text.
- [ ] **6. Develop a CLI for users to input product details and receive generated descriptions.**  
      Hint: Use a Python library like `argparse` to build the command-line interface.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the CLI to specify desired length or tone for the generated descriptions.
- Extend the CLI to manage multiple product categories or save generated descriptions to a file.
- Experiment with different training parameters or model sizes to optimize description quality.
