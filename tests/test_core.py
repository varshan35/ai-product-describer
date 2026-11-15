"""Tests for AI Product Describer milestones."""

import math

import pytest
import torch

from app.core import (
    BLOCK_SIZE,
    CORPUS,
    N_EMBD,
    CausalSelfAttention,
    MiniGPT,
    SimpleBPETokenizer,
    generate_description,
    get_batch,
    get_trained,
    load_checkpoint,
    main,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6,
    save_checkpoint,
    train_model,
)


@pytest.fixture(scope="module")
def tokenizer():
  return SimpleBPETokenizer(num_merges=150).fit(CORPUS)


@pytest.fixture(scope="module")
def tiny_trained():
  return train_model(steps=40, seed=0)


# Milestone 1: Focused Product Description Dataset
def test_milestone_1():
  """Tests that the corpus is a diverse list of unique descriptions."""
  corpus = milestone_1()
  assert isinstance(corpus, list)
  assert len(corpus) >= 50
  assert all(isinstance(s, str) and s.strip() for s in corpus)
  assert len(set(corpus)) == len(corpus)
  text = " ".join(corpus).lower()
  for word in ["leather", "wireless", "ceramic", "yoga", "garden", "puppies"]:
    assert word in text


# Milestone 2: BPE tokenizer and embeddings
def test_tokenizer_roundtrip(tokenizer):
  """Tests that decode(encode(text)) restores every corpus description."""
  for text in CORPUS:
    expected = " ".join(text.lower().split())
    assert tokenizer.decode(tokenizer.encode(text)) == expected


def test_tokenizer_has_no_unknowns_on_corpus(tokenizer):
  """Tests that fit and encode agree, so training text never maps to <unk>."""
  for text in CORPUS:
    assert tokenizer.unk_id not in tokenizer.encode(text)


def test_tokenizer_decode_does_not_split_words(tokenizer):
  """Tests the old bug where subwords were separated by spaces."""
  assert tokenizer.decode(tokenizer.encode("leather wallet")) == "leather wallet"


def test_tokenizer_merges_reduce_length(tokenizer):
  """Tests that BPE compresses text below one token per character."""
  text = CORPUS[0]
  assert len(tokenizer.encode(text)) < len(text)


def test_tokenizer_special_tokens(tokenizer):
  """Tests <bos>/<eos> handling and that decode skips them."""
  ids = tokenizer.encode("leather", add_bos=True, add_eos=True)
  assert ids[0] == tokenizer.bos_id
  assert ids[-1] == tokenizer.eos_id
  assert tokenizer.decode(ids) == "leather"


def test_tokenizer_unknown_characters(tokenizer):
  """Tests that characters never seen in training map to <unk>."""
  assert tokenizer.unk_id in tokenizer.encode("\u4f60\u597d")


def test_tokenizer_open_last_word(tokenizer):
  """Tests that a partial word is left open for the model to complete."""
  ids = tokenizer.encode("leath", close_last_word=False)
  assert not tokenizer.inverse_vocab[ids[-1]].endswith("</w>")
  assert tokenizer.decode(ids) == "leath"


def test_milestone_2():
  """Tests the tokenizer and embedding layer outputs."""
  out = milestone_2()
  n_tokens = len(out["tokenizer"].encode("leather wallet"))
  assert out["sample_out"].shape == (1, n_tokens, N_EMBD)


# Milestone 3: Causal attention and transformer block
def test_attention_is_causal():
  """Tests that changing future tokens never changes earlier outputs."""
  torch.manual_seed(0)
  attn = CausalSelfAttention(head_size=16, n_embd=N_EMBD)
  x1 = torch.randn(1, 8, N_EMBD)
  x2 = x1.clone()
  x2[:, 5:, :] = torch.randn(1, 3, N_EMBD)
  out1, out2 = attn(x1), attn(x2)
  assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-6)
  assert not torch.allclose(out1[:, 5:], out2[:, 5:])


def test_milestone_3():
  """Tests that the block preserves the input shape."""
  out = milestone_3()
  assert out["output"].shape == (1, 8, N_EMBD)


# Milestone 4: Full GPT model and training
def test_get_batch_is_shifted_by_one():
  """Tests that targets are the inputs shifted one token left."""
  data = torch.arange(500)
  x, y = get_batch(data, block_size=10, batch_size=4)
  assert x.shape == y.shape == (4, 10)
  assert torch.equal(x[:, 1:], y[:, :-1])
  assert torch.equal(y, x + 1)


def test_model_initial_loss_matches_random_guessing():
  """Tests that an untrained model's loss is close to ln(vocab size)."""
  torch.manual_seed(0)
  vocab_size = 120
  model = MiniGPT(vocab_size=vocab_size)
  x = torch.randint(vocab_size, (4, 32))
  y = torch.randint(vocab_size, (4, 32))
  logits, loss = model(x, y)
  assert logits.shape == (4, 32, vocab_size)
  assert abs(loss.item() - math.log(vocab_size)) < 0.5


def test_model_rejects_overlong_sequences():
  """Tests that sequences beyond the block size raise a clear error."""
  model = MiniGPT(vocab_size=50)
  with pytest.raises(ValueError):
    model(torch.zeros(1, BLOCK_SIZE + 1, dtype=torch.long))


def test_milestone_4_loss_decreases():
  """Tests that training reduces the loss and reports a held-out loss."""
  result = milestone_4(steps=200, seed=0)
  assert result["final_loss"] < result["initial_loss"]
  assert result["val_loss"] is not None
  assert len(result["losses"]) == 200


def test_train_model_rejects_bad_input():
  """Tests validation of training arguments."""
  with pytest.raises(ValueError):
    train_model(steps=0)
  with pytest.raises(ValueError):
    train_model(texts=[], steps=1)


def test_checkpoint_roundtrip(tiny_trained, tmp_path):
  """Tests that a saved model reproduces the same logits after loading."""
  path = tmp_path / "model.pt"
  save_checkpoint(path, tiny_trained["model"], tiny_trained["tokenizer"])
  loaded = load_checkpoint(path)

  assert loaded["tokenizer"].vocab == tiny_trained["tokenizer"].vocab
  x = torch.tensor([loaded["tokenizer"].encode("vintage leather")])
  tiny_trained["model"].eval()
  with torch.no_grad():
    expected, _ = tiny_trained["model"](x)
    actual, _ = loaded["model"](x)
  assert torch.allclose(expected, actual, atol=1e-6)


def test_get_trained_caches_to_disk(tmp_path):
  """Tests that the first call trains and saves, and the second call loads."""
  path = tmp_path / "model.pt"
  first = get_trained(path, steps=20)
  assert path.exists()
  second = get_trained(path, steps=20)
  assert second["tokenizer"].vocab == first["tokenizer"].vocab
  assert "losses" not in second


# Milestone 5: Text generation with temperature and top-k
def test_generate_starts_with_prompt(tiny_trained):
  """Tests that the output continues from the prompt."""
  text = generate_description(
      tiny_trained["model"], tiny_trained["tokenizer"],
      prompt="vintage leather", max_new_tokens=10, seed=1)
  assert isinstance(text, str)
  assert text.startswith("vintage leather")


def test_generate_is_reproducible_with_seed(tiny_trained):
  """Tests that the same seed gives the same text."""
  args = (tiny_trained["model"], tiny_trained["tokenizer"])
  a = generate_description(*args, prompt="durable", max_new_tokens=15, seed=7)
  b = generate_description(*args, prompt="durable", max_new_tokens=15, seed=7)
  assert a == b


def test_generate_greedy_is_deterministic(tiny_trained):
  """Tests that temperature 0 ignores the seed."""
  args = (tiny_trained["model"], tiny_trained["tokenizer"])
  a = generate_description(*args, prompt="compact", temperature=0, seed=1)
  b = generate_description(*args, prompt="compact", temperature=0, seed=2)
  assert a == b


def test_generate_respects_max_new_tokens(tiny_trained):
  """Tests that generation adds at most max_new_tokens tokens (so at most that many words)."""
  text = generate_description(
      tiny_trained["model"], tiny_trained["tokenizer"],
      prompt="", max_new_tokens=5, seed=3)
  assert len(text.split()) <= 5


def test_generate_accepts_empty_prompt(tiny_trained):
  """Tests that an empty prompt starts from <bos> instead of crashing."""
  text = generate_description(
      tiny_trained["model"], tiny_trained["tokenizer"],
      prompt="", max_new_tokens=5, seed=3)
  assert isinstance(text, str)


def test_generate_rejects_bad_parameters(tiny_trained):
  """Tests validation of generation settings."""
  m, t = tiny_trained["model"], tiny_trained["tokenizer"]
  with pytest.raises(ValueError):
    generate_description(m, t, max_new_tokens=0)
  with pytest.raises(ValueError):
    generate_description(m, t, temperature=-1)
  with pytest.raises(ValueError):
    generate_description(m, t, top_k=0)


def test_milestone_5(tiny_trained):
  """Tests the milestone 5 wrapper."""
  out = milestone_5(trained=tiny_trained, prompt="vintage leather",
                    max_new_tokens=8, seed=1)
  assert out["generated_text"].startswith("vintage leather")


# Milestone 6: CLI
def test_milestone_6_and_cli(tmp_path):
  """Tests that the CLI loads a saved model and prints a description."""
  path = tmp_path / "model.pt"
  get_trained(path, steps=20)

  text = milestone_6("minimalist leather", max_new_tokens=8, seed=1,
                     checkpoint=path)
  assert text.startswith("minimalist leather")

  out = []
  code = main(["wireless", "-n", "8", "--seed", "1", "--checkpoint", str(path)],
              output_fn=out.append)
  assert code == 0
  assert len(out) == 1 and out[0].startswith("wireless")


def test_cli_reports_errors(tmp_path):
  """Tests that invalid settings print an error and return 1."""
  path = tmp_path / "model.pt"
  get_trained(path, steps=20)
  out = []
  code = main(["hi", "--top-k", "0", "--checkpoint", str(path)],
              output_fn=out.append)
  assert code == 1
  assert out[0].startswith("Error")

def test_custom_product_generation(tiny_trained):
  """Tests that the model handles novel boutique product prompts."""
  text = generate_description(
      tiny_trained["model"],
      tiny_trained["tokenizer"],
      prompt="handmade ceramic",
      max_new_tokens=12,
      seed=42,
  )
  assert isinstance(text, str)
  assert "handmade" in text.lower()