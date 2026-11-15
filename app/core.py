"""AI Product Describer - Core GPT Implementation.
"""

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters for laptop CPU execution
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
BLOCK_SIZE = 96
NUM_MERGES = 150
BATCH_SIZE = 16
TRAIN_STEPS = 1500
LEARNING_RATE = 2e-3
SEED = 1337
DEVICE = "cpu"
CHECKPOINT_PATH = Path(__file__).with_name("mini_gpt.pt")

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

# =====================================================================
# MILESTONE 1: CORPUS
# =====================================================================
CORPUS = [
    # Leather goods
    "Handcrafted full-grain leather wallet with six card slots and RFID protection.",
    "Minimalist bifold leather cardholder made from vegetable-tanned Italian leather.",
    "Vintage leather messenger bag with padded laptop sleeve and antique brass buckles.",
    "Durable leather passport cover with hidden travel cash compartment and pen holder.",
    "Classic leather belt with brushed brass buckle hand-stitched for everyday durability.",
    "Premium leather watch strap featuring quick-release spring bars and soft nubuck lining.",
    "Rustic leather journal with refillable blank parchment pages and wrap-around tie.",
    "Slim leather key organizer pouch preventing pocket scratches and keeping keys silent.",
    "Artisan leather dopp kit for travel toiletries with water-resistant interior lining.",
    "Rugged leather duffle bag with reinforced handles and detachable padded shoulder strap.",
    "Compact leather coin purse with secure snap closure and textured pebble finish.",
    "Hand-stitched leather mouse pad with non-slip suede backing for home office desks.",

    # Consumer electronics
    "Wireless over-ear headphones with active noise cancellation and forty hours of battery life.",
    "Compact bluetooth speaker with waterproof housing and deep bass for beach days.",
    "Ultra-thin laptop stand made from brushed aluminum with adjustable height and cooling vents.",
    "Mechanical keyboard with hot-swappable switches, customizable backlighting, and a braided USB cable.",
    "Portable power bank with fast charging, dual USB ports, and an LED battery indicator.",
    "Smart fitness tracker monitoring heart rate, sleep quality, and daily steps with a bright touchscreen.",
    "Noise-isolating wireless earbuds with touch controls and a pocket-sized charging case.",
    "High-resolution webcam with autofocus, built-in privacy shutter, and dual noise-reducing microphones.",

    # Kitchen and dining
    "Pre-seasoned cast iron skillet that retains heat evenly and lasts for generations.",
    "Razor-sharp stainless steel chef knife with an ergonomic walnut handle and protective sheath.",
    "Insulated stainless steel water bottle that keeps drinks ice cold for twenty-four hours.",
    "Hand-glazed ceramic coffee mug with a comfortable handle and a speckled matte finish.",
    "Bamboo cutting board with deep juice grooves and a naturally antibacterial surface.",
    "Glass french press with a double-wall design for rich, full-bodied coffee at home.",
    "Nonstick ceramic baking sheet set with silicone grips that release cookies effortlessly.",
    "Stackable airtight food containers with leakproof lids for meal prep and fridge organization.",

    # Apparel and accessories
    "Lightweight merino wool sweater that regulates temperature and resists odor naturally.",
    "Breathable organic cotton t-shirt with a relaxed fit and reinforced stitched collar.",
    "Waterproof hiking jacket with taped seams, adjustable hood, and zippered underarm vents.",
    "Soft fleece-lined beanie that keeps your head warm on freezing winter mornings.",
    "Polarized aviator sunglasses with scratch-resistant lenses and a flexible titanium frame.",
    "Quick-drying running shorts with a hidden zip pocket and reflective trim for night visibility.",
    "Cozy cashmere scarf woven in a classic herringbone pattern for timeless winter style.",
    "Comfortable canvas sneakers with cushioned insoles and durable rubber outsoles for daily wear.",

    # Home and living
    "Soy wax candle hand-poured with lavender and cedarwood for a calming evening atmosphere.",
    "Plush hypoallergenic pillow with cooling gel memory foam that supports healthy neck alignment.",
    "Linen duvet cover that grows softer with every wash and keeps bedrooms cool in summer.",
    "Modern floating wall shelf crafted from solid oak with hidden mounting brackets.",
    "Woven cotton throw blanket with fringed edges that adds warmth and texture to any sofa.",
    "Ultrasonic aroma diffuser with color-changing lights and an automatic shutoff for safe sleep.",
    "Minimalist ceramic planter with a drainage hole and bamboo tray for small indoor succulents.",
    "Blackout curtains with thermal insulation that block sunlight and reduce outside noise.",

    # Outdoor and travel
    "Ultralight two-person camping tent with a rainfly, aluminum poles, and quick-setup design.",
    "Compact down sleeping bag rated for freezing nights and packable into a tiny stuff sack.",
    "Sturdy hiking backpack with ventilated back panel, hip belt, and built-in rain cover.",
    "Rechargeable camping lantern with adjustable brightness and a foldable hook for tents.",
    "Collapsible trekking poles with ergonomic cork grips and tungsten tips for rocky trails.",
    "Hardshell carry-on suitcase with spinner wheels, a built-in lock, and expandable storage.",
    "Packable travel pillow with memory foam support and a removable washable cover.",
    "Stainless steel camp cookware set that nests together and fits inside a single mesh bag.",

    # Fitness and wellness
    "Non-slip natural rubber yoga mat with alignment lines and a lightweight carrying strap.",
    "Adjustable dumbbell set that replaces fifteen weights and saves space in small apartments.",
    "Resistance band kit with five tension levels, door anchor, and a convenient travel pouch.",
    "High-density foam roller that relieves sore muscles and improves flexibility after workouts.",
    "Premium jump rope with ball-bearing handles and a tangle-free coated steel cable.",
    "Organic herbal tea sampler with soothing chamomile, peppermint, and ginger blends.",
    "Gentle plant-based shampoo bar that lathers richly and leaves hair soft without plastic waste.",
    "Hydrating facial moisturizer with hyaluronic acid and vitamin E for smooth, radiant skin.",

    # Toys, kids, and pets
    "Wooden building blocks in assorted shapes that spark creativity and develop fine motor skills.",
    "Interactive plush dinosaur with soft fabric, gentle sounds, and machine-washable stuffing.",
    "Colorful magnetic puzzle set that encourages problem solving for curious young learners.",
    "Orthopedic pet bed with a removable cover and bolstered edges for dogs of every size.",
    "Durable rope chew toy that cleans teeth and entertains energetic puppies for hours.",
    "Collapsible silicone dog bowl with a carabiner clip that is perfect for walks and road trips.",
    "Self-cleaning cat brush that gently removes loose fur and massages sensitive skin.",
    "Reflective adjustable dog leash with a padded handle for safe and comfortable evening strolls.",

    # Stationery and office
    "Refillable gel pen set in assorted colors that glides smoothly and dries instantly.",
    "Spiral-bound dotted notebook with thick acid-free paper that prevents ink from bleeding through.",
    "Ergonomic office chair with adjustable lumbar support, breathable mesh, and smooth-rolling casters.",
    "Adjustable standing desk converter that lets you switch between sitting and standing in seconds.",
    "Desk organizer made from natural walnut with compartments for pens, phones, and sticky notes.",
    "Dual-sided whiteboard with a magnetic surface and an aluminum frame for brainstorming sessions.",
    "Slim canvas laptop sleeve with soft interior padding and a water-resistant zippered pocket.",
    "Eco-friendly recycled paper planner with weekly layouts, monthly goals, and a sturdy elastic closure.",

    # Garden and tools
    "Ergonomic stainless steel garden trowel with a rust-resistant blade and cushioned grip.",
    "Self-watering herb planter that keeps basil, mint, and parsley fresh on any sunny windowsill.",
    "Cordless power drill with a lithium-ion battery, twenty-one torque settings, and a built-in work light.",
    "Heavy-duty canvas tool bag with reinforced base and multiple pockets for organized storage.",
    "Folding pruning shears with precision-ground blades and a safety lock for tidy hedges.",
    "Solar-powered garden lights that charge by day and glow softly along paths at night.",
    "Compact rechargeable handheld vacuum with strong suction and washable filters for cars and sofas.",
    "Magnetic wristband that holds screws, nails, and small bits within easy reach while you work.",
]


def milestone_1():
  """Returns the collected product description corpus."""
  return CORPUS


# =====================================================================
# MILESTONE 2: BPE TOKENIZER & EMBEDDINGS (Lessons 2 & 3)
# =====================================================================
class SimpleBPETokenizer:
  """Byte-Pair Encoding tokenizer that starts with characters and merges frequent pairs."""

  def __init__(self, num_merges=NUM_MERGES):
    self.num_merges = num_merges
    self.merges = {}
    self.vocab = {}
    self.inverse_vocab = {}

  @property
  def pad_id(self):
    return self.vocab[PAD]

  @property
  def unk_id(self):
    return self.vocab[UNK]

  @property
  def bos_id(self):
    return self.vocab[BOS]

  @property
  def eos_id(self):
    return self.vocab[EOS]

  def set_vocab(self, vocab):
    """Installs a token-to-id mapping and builds its inverse."""
    self.vocab = dict(vocab)
    self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}

  def fit(self, texts):
    word_counts = Counter()
    for text in texts:
      for word in text.lower().split():
        word_counts[" ".join(word) + " </w>"] += 1

    current_vocab = dict(word_counts)
    self.merges = {}
    for _ in range(self.num_merges):
      pairs = defaultdict(int)
      for word, freq in current_vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
          pairs[(symbols[i], symbols[i + 1])] += freq
      if not pairs:
        break
      best_pair = max(pairs, key=pairs.get)
      merged = "".join(best_pair)
      self.merges[best_pair] = merged

      # Match whole symbols only, so "e r" never matches inside "he r".
      pattern = re.compile(
          r"(?<!\S)" + re.escape(" ".join(best_pair)) + r"(?!\S)"
      )
      v_out = {}
      for word, freq in current_vocab.items():
        w_out = pattern.sub(lambda _m, merged=merged: merged, word)
        v_out[w_out] = v_out.get(w_out, 0) + freq
      current_vocab = v_out

    all_tokens = {tok for word in current_vocab for tok in word.split()}
    self.set_vocab(
        {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS + sorted(all_tokens))}
    )
    return self

  def encode(self, text, add_bos=False, add_eos=False, close_last_word=True):
    """Converts text to token ids.

    Set close_last_word=False to leave the final word open, so a model can
    complete a partial word such as "leath".
    """
    words = text.lower().split()
    tokens = []
    for w_idx, word in enumerate(words):
      symbols = list(word)
      if close_last_word or w_idx < len(words) - 1:
        symbols.append("</w>")
      for pair, merged in self.merges.items():
        i = 0
        while i < len(symbols) - 1:
          if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            symbols[i : i + 2] = [merged]
          else:
            i += 1
      tokens.extend(symbols)

    ids = [self.vocab.get(t, self.unk_id) for t in tokens]
    if add_bos:
      ids = [self.bos_id] + ids
    if add_eos:
      ids = ids + [self.eos_id]
    return ids

  def decode(self, token_ids):
    """Converts ids back to text: subwords join directly, </w> marks a space."""
    pieces = []
    for idx in token_ids:
      tok = self.inverse_vocab.get(int(idx), UNK)
      if tok in (PAD, BOS, EOS):
        continue
      pieces.append(tok)
    return " ".join("".join(pieces).replace("</w>", " ").split())


class GPTEmbedding(nn.Module):
  """Token embedding + Positional embedding (Lesson 3)."""

  def __init__(self, vocab_size, n_embd=N_EMBD, block_size=BLOCK_SIZE):
    super().__init__()
    self.tok_emb = nn.Embedding(vocab_size, n_embd)
    self.pos_emb = nn.Embedding(block_size, n_embd)

  def forward(self, idx):
    B, T = idx.shape
    pos = torch.arange(0, T, device=idx.device)
    return self.tok_emb(idx) + self.pos_emb(pos)


def milestone_2():
  """Implements BPE tokenization and generates embeddings."""
  tokenizer = SimpleBPETokenizer(num_merges=NUM_MERGES).fit(CORPUS)
  emb = GPTEmbedding(vocab_size=len(tokenizer.vocab))
  sample_tokens = torch.tensor([tokenizer.encode("leather wallet")])
  output = emb(sample_tokens)
  return {"tokenizer": tokenizer, "embeddings": emb, "sample_out": output}


# =====================================================================
# MILESTONE 3: CAUSAL ATTENTION & TRANSFORMER BLOCK (Lessons 4, 5, 6)
# =====================================================================
class CausalSelfAttention(nn.Module):
  """Single head with scaled dot-product attention and tril causal mask (Lessons 4 & 5)."""

  def __init__(self, head_size=16, n_embd=N_EMBD, block_size=BLOCK_SIZE):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    # The No-Peeking Rulebook (lower triangular matrix)
    self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)  # (B, T, head_size)
    q = self.query(x)  # (B, T, head_size)
    v = self.value(x)  # (B, T, head_size)

    # Attention scores: Q @ K.T / sqrt(head_size)
    wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1]))
    # Mask future tokens with -inf so they receive 0% attention after softmax
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    return wei @ v  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
  """Multiple heads of self-attention in parallel (Lesson 6)."""

  def __init__(self, num_heads, head_size, n_embd=N_EMBD, block_size=BLOCK_SIZE):
    super().__init__()
    self.heads = nn.ModuleList(
        [CausalSelfAttention(head_size, n_embd, block_size) for _ in range(num_heads)]
    )
    self.proj = nn.Linear(num_heads * head_size, n_embd)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    return self.proj(out)


class FeedForward(nn.Module):
  """A simple per-token MLP: expand 4x -> ReLU -> project back (Lesson 6)."""

  def __init__(self, n_embd=N_EMBD):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd)
    )

  def forward(self, x):
    return self.net(x)


class TransformerBlock(nn.Module):
  """Pre-norm residual transformer block (Lesson 6)."""

  def __init__(self, n_embd=N_EMBD, n_head=N_HEAD, block_size=BLOCK_SIZE):
    super().__init__()
    if n_embd % n_head != 0:
      raise ValueError("n_embd must be divisible by n_head.")
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    # Residual connections: x + sublayer(norm(x))
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


def milestone_3():
  """Constructs causal attention and the transformer block."""
  attn = CausalSelfAttention(head_size=16, n_embd=N_EMBD)
  block = TransformerBlock(n_embd=N_EMBD, n_head=N_HEAD)
  sample_input = torch.randn(1, 8, N_EMBD)
  return {"attention": attn, "block": block, "output": block(sample_input)}


# =====================================================================
# MILESTONE 4: FULL GPT MODEL & TRAINING (Lessons 1 & 7)
# =====================================================================
class MiniGPT(nn.Module):
  """Decoder-only transformer language model."""

  def __init__(
      self,
      vocab_size,
      n_embd=N_EMBD,
      n_head=N_HEAD,
      n_layer=N_LAYER,
      block_size=BLOCK_SIZE,
  ):
    super().__init__()
    self.block_size = block_size
    self.config = {
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
    }
    self.embedding = GPTEmbedding(vocab_size, n_embd, block_size)
    self.blocks = nn.Sequential(
        *[TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)]
    )
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.apply(self._init_weights)

  @staticmethod
  def _init_weights(module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    if T > self.block_size:
      raise ValueError(f"Sequence length {T} exceeds block size {self.block_size}.")
    x = self.embedding(idx)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    loss = None
    if targets is not None:
      # Cross-entropy next-token loss (Lesson 1)
      B, T, C = logits.shape
      loss = F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T))

    return logits, loss


def build_dataset(tokenizer, texts):
  """Encodes each description as <bos> ... <eos> and joins them into one stream."""
  ids = []
  for text in texts:
    ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
  return torch.tensor(ids, dtype=torch.long)


def get_batch(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
  """Samples random windows; y is x shifted one token to the left."""
  block = min(block_size, len(data) - 1)
  if block < 1:
    raise ValueError("Not enough data to build a batch.")
  ix = torch.randint(len(data) - block, (batch_size,))
  x = torch.stack([data[i : i + block] for i in ix])
  y = torch.stack([data[i + 1 : i + block + 1] for i in ix])
  return x.to(DEVICE), y.to(DEVICE)


def estimate_loss(model, data, batches=10, batch_size=BATCH_SIZE):
  """Average loss over several random batches, without updating weights."""
  model.eval()
  losses = []
  with torch.no_grad():
    for _ in range(batches):
      x, y = get_batch(data, model.block_size, batch_size)
      _, loss = model(x, y)
      losses.append(loss.item())
  model.train()
  return sum(losses) / len(losses)


def train_model(
    texts=None,
    steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    num_merges=NUM_MERGES,
    seed=SEED,
):
  """Trains the mini-GPT and reports start, end, and held-out loss."""
  if steps < 1:
    raise ValueError("steps must be at least 1.")
  texts = list(CORPUS if texts is None else texts)
  if not texts:
    raise ValueError("No training texts provided.")

  torch.manual_seed(seed)
  rng = random.Random(seed)

  tokenizer = SimpleBPETokenizer(num_merges=num_merges).fit(texts)

  shuffled = texts[:]
  rng.shuffle(shuffled)
  n_val = max(1, len(shuffled) // 10) if len(shuffled) >= 10 else 0
  train_texts = shuffled[: len(shuffled) - n_val]
  val_texts = shuffled[len(shuffled) - n_val :]
  train_data = build_dataset(tokenizer, train_texts)
  val_data = build_dataset(tokenizer, val_texts) if val_texts else None

  model = MiniGPT(vocab_size=len(tokenizer.vocab)).to(DEVICE)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

  model.train()
  losses = []
  for _ in range(steps):
    x, y = get_batch(train_data, model.block_size, batch_size)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())

  window = min(20, len(losses))
  return {
      "model": model,
      "tokenizer": tokenizer,
      "initial_loss": sum(losses[:window]) / window,
      "final_loss": sum(losses[-window:]) / window,
      "val_loss": (
          estimate_loss(model, val_data, batch_size=batch_size)
          if val_data is not None
          else None
      ),
      "losses": losses,
  }


def save_checkpoint(path, model, tokenizer):
  """Saves weights, model config, and tokenizer so training happens only once."""
  torch.save(
      {
          "state_dict": model.state_dict(),
          "config": model.config,
          "merges": [list(pair) for pair in tokenizer.merges],
          "vocab": tokenizer.vocab,
      },
      path,
  )


def load_checkpoint(path):
  """Restores a model and tokenizer saved by save_checkpoint."""
  ckpt = torch.load(path, map_location=DEVICE)
  tokenizer = SimpleBPETokenizer(num_merges=len(ckpt["merges"]))
  tokenizer.merges = {tuple(pair): "".join(pair) for pair in ckpt["merges"]}
  tokenizer.set_vocab(ckpt["vocab"])
  model = MiniGPT(**ckpt["config"]).to(DEVICE)
  model.load_state_dict(ckpt["state_dict"])
  model.eval()
  return {"model": model, "tokenizer": tokenizer}


def get_trained(path=CHECKPOINT_PATH, retrain=False, **train_kwargs):
  """Loads a saved model if one exists, otherwise trains and saves one."""
  path = Path(path)
  if not retrain and path.exists():
    try:
      return load_checkpoint(path)
    except Exception:  # Stale or corrupt checkpoint: fall back to retraining.
      pass

  trained = train_model(**train_kwargs)
  try:
    save_checkpoint(path, trained["model"], trained["tokenizer"])
  except OSError:
    pass
  return trained


def milestone_4(steps=TRAIN_STEPS, seed=SEED):
  """Trains the mini-GPT model on the product description dataset."""
  return train_model(steps=steps, seed=seed)


# =====================================================================
# MILESTONE 5: TEXT GENERATION WITH TEMPERATURE & TOP-K (Lesson 8)
# =====================================================================
def generate_description(
    model,
    tokenizer,
    prompt="handcrafted leather",
    max_new_tokens=60,
    temperature=0.8,
    top_k=5,
    seed=None,
    partial_last_word=False,
):
  """Autoregressively generates a product description.

  Generation stops at <eos> or after max_new_tokens. temperature=0 is greedy.
  """
  if max_new_tokens < 1:
    raise ValueError("max_new_tokens must be at least 1.")
  if temperature < 0:
    raise ValueError("temperature must not be negative.")
  if top_k is not None and top_k < 1:
    raise ValueError("top_k must be at least 1.")

  model.eval()
  generator = None
  if seed is not None:
    generator = torch.Generator().manual_seed(seed)

  tokens = [tokenizer.bos_id] + tokenizer.encode(
      prompt, close_last_word=not partial_last_word
  )
  idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
  blocked = [tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id]

  for _ in range(max_new_tokens):
    idx_cond = idx[:, -model.block_size :]
    with torch.no_grad():
      logits, _ = model(idx_cond)
    logits = logits[:, -1, :]
    logits[:, blocked] = float("-inf")

    if temperature == 0:
      next_token = torch.argmax(logits, dim=-1, keepdim=True)
    else:
      logits = logits / temperature
      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
      probs = F.softmax(logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1, generator=generator)

    if next_token.item() == tokenizer.eos_id:
      break
    idx = torch.cat((idx, next_token), dim=1)

  return tokenizer.decode(idx[0].tolist())


def milestone_5(trained=None, prompt="vintage leather", max_new_tokens=40, seed=None):
  """Generates a sample product description from a prompt."""
  trained = trained if trained is not None else get_trained()
  generated = generate_description(
      trained["model"],
      trained["tokenizer"],
      prompt=prompt,
      max_new_tokens=max_new_tokens,
      seed=seed,
  )
  return {"generated_text": generated}


# =====================================================================
# MILESTONE 6: CLI INTERFACE (Lesson 9)
# =====================================================================
def milestone_6(
    prompt="minimalist leather",
    max_new_tokens=60,
    temperature=0.8,
    top_k=5,
    seed=None,
    retrain=False,
    checkpoint=CHECKPOINT_PATH,
):
  """CLI entry-point logic to generate product descriptions."""
  trained = get_trained(checkpoint, retrain=retrain)
  return generate_description(
      trained["model"],
      trained["tokenizer"],
      prompt=prompt,
      max_new_tokens=max_new_tokens,
      temperature=temperature,
      top_k=top_k,
      seed=seed,
  )


def build_parser():
  parser = argparse.ArgumentParser(
      description="Generate product descriptions with a small GPT trained from scratch."
  )
  parser.add_argument("prompt", nargs="?", default="minimalist leather",
                      help="starting words for the description")
  parser.add_argument("-n", "--max-new-tokens", type=int, default=60)
  parser.add_argument("-t", "--temperature", type=float, default=0.8,
                      help="0 is greedy; higher is more varied")
  parser.add_argument("--top-k", type=int, default=5)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--retrain", action="store_true",
                      help="ignore any saved model and train a new one")
  parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH),
                      help="where the trained model is saved and loaded")
  return parser


def main(argv=None, output_fn=print):
  """Runs the CLI and returns an exit code."""
  args = build_parser().parse_args(argv)
  try:
    output_fn(
        milestone_6(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            retrain=args.retrain,
            checkpoint=args.checkpoint,
        )
    )
  except ValueError as err:
    output_fn(f"Error: {err}")
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())