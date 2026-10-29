"""
Single source of truth for character-level GPT architecture and experiment utilities.
Implements Karpathy's nanoGPT with configurable hyperparameters, reproducible training,
exact timing, comparable validation metrics, and text generation.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Hyperparameters according to teacher's specification."""
    vocab_size: int = 133
    batch_size: int = 32
    block_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 20
    learning_rate: float = 3e-4
    n_embd: int = 384
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_corpus(
    filepath: str = "input.txt",
    encoding: str = "utf-8",
    train_ratio: float = 0.9,
) -> Dict[str, Any]:
    """Loads corpus, computes SHA-256, creates vocabulary, and splits into train/val."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode(encoding)

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return {
        "raw_bytes": len(raw_bytes),
        "sha256": sha256,
        "text": text,
        "chars": chars,
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "encode": encode,
        "decode": decode,
        "data": data,
        "train_data": train_data,
        "val_data": val_data,
    }


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples random batch of shape (batch_size, block_size)."""
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, hs)
        out = wei @ v      # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_size: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)
        ])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """A simple linear layer followed by non-linearity."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Alias for conventional spelling
FeedForward = FeedFoward


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Character-level GPT Language Model."""

    def __init__(
        self,
        vocab_size: int = 133,
        n_embd: int = 384,
        block_size: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.1,
        config: Optional[GPTConfig] = None,
    ):
        super().__init__()
        if config is not None:
            vocab_size = config.vocab_size
            n_embd = config.n_embd
            block_size = config.block_size
            n_head = config.n_head
            n_layer = config.n_layer
            dropout = config.dropout

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)     # (B, T, C)
        x = self.ln_f(x)       # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature and top-k filtering.
        In eval mode without gradients.
        """
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature <= 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=generator)

            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()
        return idx


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    """Estimates batch-wise loss on train and val splits without affecting training state."""
    was_training = model.training
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device, generator=generator)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = float(losses.mean().item())
    if was_training:
        model.train()
    return out


@torch.no_grad()
def evaluate_fixed_targets(
    model: nn.Module,
    split_data: torch.Tensor,
    target_indices: List[int],
    block_size: int,
    batch_size: int = 32,
    device: str = "cuda",
) -> float:
    """
    Evaluates cross-entropy loss on identical target characters in split_data.
    Every target position has at least 128 preceding tokens.
    Model receives split_data[pos - block_size : pos] and predicts split_data[pos].
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = len(target_indices)

    for i in range(0, total_samples, batch_size):
        batch_targets = target_indices[i : i + batch_size]
        x = torch.stack([split_data[pos - block_size : pos] for pos in batch_targets]).to(device)
        y = torch.tensor([split_data[pos] for pos in batch_targets], dtype=torch.long, device=device)

        logits, _ = model(x)
        last_logits = logits[:, -1, :]
        loss = F.cross_entropy(last_logits, y, reduction="sum")
        total_loss += loss.item()

    if was_training:
        model.train()
    return total_loss / total_samples


def validate_prompt(prompt: str, stoi: Dict[str, int]) -> Tuple[bool, List[str]]:
    """Checks whether all characters in prompt exist in vocabulary."""
    missing = [c for c in prompt if c not in stoi]
    return len(missing) == 0, sorted(list(set(missing)))


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    config: GPTConfig,
    chars: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves model weights, config, vocabulary and metadata."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
        "chars": chars,
        "metadata": metadata or {},
    }
    torch.save(state, path)


def load_checkpoint(
    filepath: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads model from checkpoint along with config and tokenizer."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = checkpoint["config"]
    config = GPTConfig(**cfg_dict)

    model = GPTLanguageModel(config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    chars = checkpoint["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    return {
        "model": model,
        "config": config,
        "chars": chars,
        "stoi": stoi,
        "itos": itos,
        "encode": encode,
        "decode": decode,
        "metadata": checkpoint.get("metadata", {}),
    }


def run_microbenchmark(device: str, repeats: int = 4) -> Dict[str, Any]:
    """
    Performs ABBA microbenchmark to isolate forward+backward+optimizer step time
    and measure GPU throughput without altering the main training models.
    """
    results = {"context_64": [], "context_128": []}

    # Dummy models for benchmark
    m64 = GPTLanguageModel(block_size=64).to(device)
    m128 = GPTLanguageModel(block_size=128).to(device)
    opt64 = torch.optim.AdamW(m64.parameters(), lr=3e-4)
    opt128 = torch.optim.AdamW(m128.parameters(), lr=3e-4)

    # Pre-allocated tensors on GPU
    x64 = torch.randint(0, 133, (32, 64), device=device)
    y64 = torch.randint(0, 133, (32, 64), device=device)
    x128 = torch.randint(0, 133, (32, 128), device=device)
    y128 = torch.randint(0, 133, (32, 128), device=device)

    # Warmup
    for _ in range(15):
        opt64.zero_grad(set_to_none=True)
        _, l64 = m64(x64, y64)
        l64.backward()
        opt64.step()
        opt128.zero_grad(set_to_none=True)
        _, l128 = m128(x128, y128)
        l128.backward()
        opt128.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sequence = ["context_64", "context_128", "context_128", "context_64"] * repeats
    n_steps = 40

    for mode in sequence:
        m = m64 if mode == "context_64" else m128
        opt = opt64 if mode == "context_64" else opt128
        x = x64 if mode == "context_64" else x128
        y = y64 if mode == "context_64" else y128
        bs = 64 if mode == "context_64" else 128

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            _, loss = m(x, y)
            loss.backward()
            opt.step()

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            gpu_time_ms = start_event.elapsed_time(end_event) / n_steps
        else:
            gpu_time_ms = None

        wall_time_ms = (time.perf_counter() - t0) * 1000.0 / n_steps
        throughput = (32 * bs) / (wall_time_ms / 1000.0)

        results[mode].append({
            "wall_ms_per_step": wall_time_ms,
            "gpu_ms_per_step": gpu_time_ms,
            "tokens_per_sec": throughput,
        })

    summary = {}
    for mode in ["context_64", "context_128"]:
        walls = [r["wall_ms_per_step"] for r in results[mode]]
        tps = [r["tokens_per_sec"] for r in results[mode]]
        summary[mode] = {
            "wall_ms_median": float(np.median(walls)),
            "wall_ms_std": float(np.std(walls)),
            "tokens_per_sec_median": float(np.median(tps)),
            "tokens_per_sec_std": float(np.std(tps)),
            "raw_measurements": results[mode],
        }

    return summary


def train_single_model(
    config: GPTConfig,
    corpus: Dict[str, Any],
    seed: int,
    fixed_val_targets: List[int],
    fixed_train_targets: List[int],
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Trains a single model instance with exact pure training time measurement,
    isolated RNGs, comparable fixed-target evaluation on train and val,
    and automatic checkpoint saving (step 2500, best by Val NLL, and final step 5000).
    """
    label = f"context_{config.block_size}_seed{seed}"
    if verbose:
        print(f"\n[{label}] Начало обучения (seed={seed}, block_size={config.block_size}, max_iters={config.max_iters})")

    # Master seed and isolated generators
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_rng = torch.Generator(device="cpu").manual_seed(seed)
    eval_rng = torch.Generator(device="cpu").manual_seed(42)

    device = config.device
    model = GPTLanguageModel(config=config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    pos_emb_params = config.block_size * config.n_embd

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    history = []
    pure_training_time = 0.0
    wall_start = time.perf_counter()
    tokens_per_step = config.batch_size * config.block_size

    # Step 0 evaluation
    init_batch_losses = estimate_loss(
        model,
        corpus["train_data"],
        corpus["val_data"],
        config.batch_size,
        config.block_size,
        config.eval_iters,
        device,
        generator=eval_rng,
    )
    init_val_nll = evaluate_fixed_targets(
        model, corpus["val_data"], fixed_val_targets, config.block_size, 32, device
    )
    init_train_nll = evaluate_fixed_targets(
        model, corpus["train_data"], fixed_train_targets, config.block_size, 32, device
    )

    history.append({
        "step": 0,
        "train_loss_batch": init_batch_losses["train"],
        "val_loss_batch": init_batch_losses["val"],
        "train_nll_comp": init_train_nll,
        "val_nll_comp": init_val_nll,
        "processed_tokens": 0,
        "pure_time": 0.0,
    })

    best_val_nll = init_val_nll
    best_step = 0
    val_nll_at_2500 = None

    if save_dir is not None:
        save_checkpoint(
            save_dir / f"checkpoint_{config.block_size}_seed{seed}_best.pt",
            model, config, corpus["chars"], metadata={"step": 0, "val_nll_comp": best_val_nll}
        )

    num_intervals = config.max_iters // config.eval_interval

    for interval_idx in range(num_intervals):
        start_step = interval_idx * config.eval_interval
        end_step = (interval_idx + 1) * config.eval_interval

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_chunk_start = time.perf_counter()

        model.train()
        for _ in range(start_step, end_step):
            xb, yb = get_batch(
                corpus["train_data"],
                config.block_size,
                config.batch_size,
                device,
                generator=train_rng,
            )
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pure_training_time += (time.perf_counter() - t_chunk_start)

        # Evaluation outside timed training block
        batch_losses = estimate_loss(
            model,
            corpus["train_data"],
            corpus["val_data"],
            config.batch_size,
            config.block_size,
            config.eval_iters,
            device,
            generator=eval_rng,
        )
        val_nll_comp = evaluate_fixed_targets(
            model, corpus["val_data"], fixed_val_targets, config.block_size, 32, device
        )
        train_nll_comp = evaluate_fixed_targets(
            model, corpus["train_data"], fixed_train_targets, config.block_size, 32, device
        )

        total_tokens = end_step * tokens_per_step
        history.append({
            "step": end_step,
            "train_loss_batch": batch_losses["train"],
            "val_loss_batch": batch_losses["val"],
            "train_nll_comp": train_nll_comp,
            "val_nll_comp": val_nll_comp,
            "processed_tokens": total_tokens,
            "pure_time": pure_training_time,
        })

        if end_step == 2500 and config.block_size == 128:
            val_nll_at_2500 = val_nll_comp
            if save_dir is not None:
                save_checkpoint(
                    save_dir / f"checkpoint_128_seed{seed}_step2500.pt",
                    model, config, corpus["chars"],
                    metadata={"step": 2500, "val_nll_comp": val_nll_comp, "processed_tokens": total_tokens}
                )

        if val_nll_comp < best_val_nll:
            best_val_nll = val_nll_comp
            best_step = end_step
            if save_dir is not None:
                save_checkpoint(
                    save_dir / f"checkpoint_{config.block_size}_seed{seed}_best.pt",
                    model, config, corpus["chars"],
                    metadata={"step": best_step, "val_nll_comp": best_val_nll, "processed_tokens": total_tokens}
                )

        if verbose and (end_step % 1000 == 0 or end_step == config.max_iters):
            print(f"[{label}] шаг {end_step:4d}/{config.max_iters}: "
                  f"Train NLL={train_nll_comp:.4f}, Val NLL={val_nll_comp:.4f} "
                  f"(чистое время: {pure_training_time:.1f}с)")

    total_experiment_time = time.perf_counter() - wall_start
    final_val_nll = history[-1]["val_nll_comp"]
    final_train_nll = history[-1]["train_nll_comp"]
    nll_degradation = final_val_nll - best_val_nll

    if save_dir is not None:
        save_checkpoint(
            save_dir / f"checkpoint_{config.block_size}_seed{seed}_final.pt",
            model, config, corpus["chars"],
            metadata={"step": config.max_iters, "val_nll_comp": final_val_nll, "processed_tokens": config.max_iters * tokens_per_step}
        )

    sec_per_step = pure_training_time / config.max_iters
    ms_per_step = sec_per_step * 1000.0
    throughput = (config.max_iters * tokens_per_step) / pure_training_time

    if verbose:
        print(f"[{label}] Завершено: финал Val NLL={final_val_nll:.4f}, лучший={best_val_nll:.4f} (шаг {best_step}), деградация={nll_degradation:+.4f}")
        print(f"[{label}] Чистое время: {pure_training_time:.1f}с, время шага: {ms_per_step:.2f}мс, скорость: {throughput:.1f} симв/с")

    return {
        "label": label,
        "seed": seed,
        "block_size": config.block_size,
        "param_count": param_count,
        "pos_emb_params": pos_emb_params,
        "max_iters": config.max_iters,
        "processed_tokens": config.max_iters * tokens_per_step,
        "pure_training_seconds": pure_training_time,
        "total_experiment_seconds": total_experiment_time,
        "ms_per_step": ms_per_step,
        "throughput_chars_per_sec": throughput,
        "final_train_loss_batch": history[-1]["train_loss_batch"],
        "final_val_loss_batch": history[-1]["val_loss_batch"],
        "final_train_nll_comp": final_train_nll,
        "final_val_nll_comp": final_val_nll,
        "best_val_nll_comp": best_val_nll,
        "best_step": best_step,
        "nll_degradation": nll_degradation,
        "val_nll_at_2500": val_nll_at_2500,
        "history": history,
        "model": model,
        "config": config,
    }


def generate_samples_for_prompts(
    models: Dict[str, nn.Module],
    stoi: Dict[str, int],
    itos: Dict[int, str],
    prompts: List[str],
    device: str,
    temperature: float = 0.8,
    top_k: int = 40,
    max_new_tokens: int = 400,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generates continuations from identical Russian prompts with identical seed."""
    results = []

    for prompt in prompts:
        valid, missing = validate_prompt(prompt, stoi)
        if not valid:
            raise ValueError(f"Prompt '{prompt}' contains characters not in vocabulary: {missing}")

        prompt_indices = [stoi[c] for c in prompt]
        prompt_tensor = torch.tensor([prompt_indices], dtype=torch.long, device=device)

        row = {
            "prompt": prompt,
            "seed": seed,
            "temperature": temperature,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
        }
        for label, model in models.items():
            gen_rng = torch.Generator(device=device if device == "cpu" else "cuda")
            gen_rng.manual_seed(seed)
            out_tokens = model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                generator=gen_rng,
            )
            continuation = "".join([itos[i] for i in out_tokens[0].tolist()])
            row[label] = continuation

        results.append(row)

    return results


if __name__ == "__main__":
    # Standalone execution: trains baseline model (block_size=64) on Pushkin corpus
    torch.manual_seed(1337)
    corpus = load_corpus("input.txt")
    print(f"Loaded corpus: {len(corpus['text'])} characters, {corpus['vocab_size']} unique characters.")
    print(f"Corpus SHA-256: {corpus['sha256']}")

    config = GPTConfig(
        vocab_size=corpus["vocab_size"],
        batch_size=32,
        block_size=64,
        max_iters=5000,
        eval_interval=500,
        eval_iters=20,
        learning_rate=3e-4,
        n_embd=384,
        n_head=4,
        n_layer=2,
        dropout=0.1,
    )

    device = config.device
    model = GPTLanguageModel(config=config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    t0 = time.perf_counter()
    for iter_num in range(config.max_iters):
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            losses = estimate_loss(
                model,
                corpus["train_data"],
                corpus["val_data"],
                config.batch_size,
                config.block_size,
                config.eval_iters,
                device,
            )
            print(f"step {iter_num:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(corpus["train_data"], config.block_size, config.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.perf_counter() - t0
    print(f"Training completed in {total_time:.2f} s")

    # Sample generation
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    print("--- Sample generation ---")
    print(corpus["decode"](generated[0].tolist()))
