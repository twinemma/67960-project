from copy import deepcopy
from datasets import load_dataset
from datetime import datetime
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from typing import Dict, List, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
print("✅ Setup complete!")

tinystoriesdata = load_dataset(
    "roneneldan/TinyStories"
).shuffle(seed=42)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# =============================================================================
# DATASETS: Different tasks to probe transformer behavior
# =============================================================================

class CopyDataset(Dataset):
    """Simple copy - diagonal attention (EASY)"""
    def __init__(self, seq_len=32, vocab_size=16, dataset_size=4096):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dataset_size = dataset_size
        self.data = torch.randint(1, vocab_size, (dataset_size, seq_len), dtype=torch.long)

    def __len__(self): return self.dataset_size
    def __getitem__(self, idx): return self.data[idx].clone(), self.data[idx].clone()


class ReverseDataset(Dataset):
    """Reverse sequence - anti-diagonal attention (MEDIUM)"""
    def __init__(self, seq_len=32, vocab_size=16, dataset_size=4096):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dataset_size = dataset_size
        self.data = torch.randint(1, vocab_size, (dataset_size, seq_len), dtype=torch.long)

    def __len__(self): return self.dataset_size
    def __getitem__(self, idx):
        x = self.data[idx].clone()
        y = torch.flip(x, dims=[0])
        return x, y


class SortDataset(Dataset):
    """Sort sequence - global reasoning (HARD)"""
    def __init__(self, seq_len=32, vocab_size=16, dataset_size=4096):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dataset_size = dataset_size
        self.data = torch.randint(1, vocab_size, (dataset_size, seq_len), dtype=torch.long)

    def __len__(self): return self.dataset_size
    def __getitem__(self, idx):
        x = self.data[idx].clone()
        y, _ = torch.sort(x)
        return x, y


class TextPredictionDataset(Dataset):
    """Next-token prediction with local dependencies"""
    def __init__(self, seq_len=64, dataset_size=4096):
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        vocab_size = 32
        self.vocab_size = vocab_size
        self.data = []

        for _ in range(dataset_size):
            seq = [random.randint(0, vocab_size-1)]
            for i in range(1, seq_len):
                if random.random() < 0.5:  # 50% local dependency
                    seq.append((seq[-1] + random.randint(1, 5)) % vocab_size)
                else:
                    seq.append(random.randint(0, vocab_size-1))
            self.data.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self): return self.dataset_size
    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1].clone(), seq[1:].clone()  # Input: [:-1], Target: [1:]


class ImageClassificationDataset(Dataset):
    """Flattened image patches for classification"""
    def __init__(self, img_size=8, num_classes=10, dataset_size=4096):
        self.img_size = img_size
        self.num_classes = num_classes
        self.dataset_size = dataset_size
        self.seq_len = img_size * img_size
        self.vocab_size = 256
        self.data = []

        for _ in range(dataset_size):
            label = random.randint(0, num_classes - 1)
            img = torch.zeros(img_size, img_size, dtype=torch.long)

            # Class-specific patterns
            if label < 5:
                img[:, label] = 200  # Vertical
            else:
                img[label - 5, :] = 200  # Horizontal

            # Add noise
            noise = torch.randint(0, 50, (img_size, img_size), dtype=torch.long)
            img = torch.clamp(img + noise, 0, 255)
            self.data.append((img.flatten(), label))

    def __len__(self): return self.dataset_size
    def __getitem__(self, idx): return self.data[idx]


class TinyStoriesDataset(Dataset):
    def __init__(self, seq_len=32, split="train", dataset_size=4096):
        raw = tinystoriesdata[split].select(range(dataset_size))

        self.chunks = []

        for story in raw["text"]:
            ids = tokenizer.encode(story, add_special_tokens=False)

            # break each story into its own chunks
            for i in range(0, len(ids) - seq_len - 1, seq_len):
                chunk = ids[i:i+seq_len+1]
                self.chunks.append(chunk)

        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.chunks[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]


print("✅ Datasets defined:")
print("   - Copy (easy, diagonal attention)")
print("   - Reverse (medium, anti-diagonal)")
print("   - Sort (hard, global reasoning)")
print("   - Text (language modeling)")
print("   - Image (classification)")

DATASET_TYPES = {
    "copy": CopyDataset,
    "reverse": ReverseDataset,
    "sort": SortDataset,
    "text": TextPredictionDataset,
    "image": ImageClassificationDataset,
    "tiny_stories": TinyStoriesDataset
}

# =============================================================================
# Training Loss Helper Functions
# =============================================================================

def logdetreg_loss_per_layer(H, eps=1e-4):
    # H: (B, T, d)
    B, T, d = H.shape
    X = H.reshape(B*T, d)
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.T @ X) / (X.shape[0] - 1)
    cov = cov + eps * torch.eye(d, device=H.device)
    return -torch.logdet(cov)

def erank_loss_per_layer(H, eps=1e-12):
    B, T, d = H.shape
    X = H.reshape(B*T, d)
    s = torch.linalg.svdvals(X)
    er = (s.sum()**2) / ( (s*s).sum() + eps )
    return -er

def orth_loss_per_matrix(W):
    WtW = W.t() @ W
    I = torch.eye(WtW.shape[0], device=W.device)
    return ((WtW - I)**2).sum()

def spectral_norm_loss_per_matrix(W):
    s = torch.linalg.svdvals(W)
    return s[0]

# =============================================================================
# Training Loss Functions
# =============================================================================

# The main loss
crossentropyloss = torch.nn.CrossEntropyLoss()
def cross_entropy(logits, yb, **kwargs):
    task_type = kwargs["task_type"]
    if task_type in ['seq2seq', 'causal_lm']:
        B, T, V = logits.shape
        loss = crossentropyloss(logits.view(B*T, V), yb.view(B*T))
    else:
        loss = crossentropyloss(logits, yb)
    return loss

# The losses for trying to increase the rank of the matrices
def logdetregularizer(logits, yb, **kwargs):
    loss = 0
    for activation in kwargs["activations"]:
        loss += logdetreg_loss_per_layer(activation)
    return loss

def erank_loss(logits, yb, **kwargs):
    loss = 0
    for activation in kwargs["activations"]:
        loss += erank_loss_per_layer(activation)
    return loss

def orthogonal_loss(logits, yb, **kwargs):
    layers = kwargs["layers"]
    loss = 0
    for layer in layers:
        loss += orth_loss_per_matrix(layer["Wq"])
        loss += orth_loss_per_matrix(layer["Wk"])
    return loss

def spectral_norm_loss(logits, yb, **kwargs):
    layers = kwargs["layers"]
    loss = 0
    for layer in layers:
        loss += spectral_norm_loss_per_matrix(layer["Wq"])
        loss += spectral_norm_loss_per_matrix(layer["Wk"])
    return loss

# =============================================================================
# Schedulers
# =============================================================================

class ResAlphaScheduler:
    """
    Scheduler for ResNet residual connection weights
    """
    def __init__(self, alpha=1, timesteps=[]):
        self.alpha = alpha
        self.count = 0
        self.timesteps = timesteps
        if len(timesteps) != 0:
            self.increment = self.alpha / len(timesteps)

    def get_alpha(self):
        if self.count in self.timesteps:
            self.alpha -= self.increment
        self.count += 1
        return self.alpha


class NoiseScheduler:
    """
    Scheduler for injecting Gaussian noise during training
    noise_std(t) is linearly annealed from initial_std → final_std
    over timesteps.
    """
    def __init__(self, initial_std=0., final_std=0.0, timesteps=1000):
        self.initial_std = initial_std
        self.final_std = final_std
        self.timesteps = timesteps
        self.count = 0

    def get_std(self):
        if self.count >= self.timesteps:
            return self.final_std
        ratio = self.count / self.timesteps
        std = (1 - ratio) * self.initial_std + ratio * self.final_std
        self.count += 1
        return std


class TrainingScheduler:
    """
    Scheduler for different losses during training
    Expects functions_at_times to be a list of tuples where the first element
    is a starting timestep, the second is an ending timestep, and the third
    element is a function (or None)
    """
    def __init__(self, which_layers="all", base_func=cross_entropy, functions_at_times=[]):
        self.base_func = base_func
        self.added_funcs = functions_at_times
        self.count = 0
        self.which_layers = which_layers

    def _preprocess_layers(self, layers):
        increment = int(len(layers) / 3)
        if self.which_layers == "first":
            return layers[:increment]
        elif self.which_layers == "middle":
            return layers[increment:2*increment]
        elif self.which_layers == "last":
            return layers[-increment:]
        else:
            return layers

    def call_function(self, logits, target, increase_count=True, **kwargs):
        value = self.base_func(logits, target, **kwargs)
        for start, end, func in self.added_funcs:
            if start <= self.count <= end and func is not None:
                value += func(logits, target, **kwargs)

        if increase_count:
            self.count += 1
        return value


# =============================================================================
# TRANSFORMER ARCHITECTURES
# =============================================================================

class MultiHeadAttentionWithCapture(nn.Module):
    """Custom MHA that captures attention weights"""
    def __init__(self, d_model, nhead, dropout=0.1, add_noise=NoiseScheduler()):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.add_noise = add_noise

    def forward(self, query, key, value, attn_mask=None):
        B, T, C = query.shape

        # Project and reshape to (B, nhead, T, head_dim)
        Q = self.q_proj(query).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        if self.add_noise:
            Q += torch.randn_like(Q)
            K += torch.randn_like(K)
        V = self.v_proj(value).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)  # (B, nhead, T, T)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)

        return output, attn_weights

    def weight_matrices(self):
        return {
            "Wq": self.q_proj.weight.detach().cpu().numpy(),
            "Wk": self.k_proj.weight.detach().cpu().numpy(),
            "Wv": self.v_proj.weight.detach().cpu().numpy(),
            "Wo": self.out_proj.weight.detach().cpu().numpy(),
        }

class TransformerBlock(nn.Module):
    """Single transformer layer with attention capture"""
    def __init__(
            self, d_model, nhead, dim_feedforward, dropout=0.1,
            causal=False, add_noise=NoiseScheduler(), alpha=ResAlphaScheduler()):
        super().__init__()
        self.attn = MultiHeadAttentionWithCapture(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.last_attn_weights = None
        self.last_activations = None
        self.causal = causal
        self.add_noise = add_noise
        self.alpha = alpha

    def forward(self, x):
        if not self.causal:
            attn_out, attn_weights = self.attn(x, x, x)
        else:
            causal_mask = torch.triu(
                torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device),
                diagonal=1
            )
            attn_out, attn_weights = self.attn(x, x, x, causal_mask)
        self.last_attn_weights = attn_weights.detach().cpu().numpy()
        x = self.norm1(self.alpha.get_alpha() * x + attn_out)
        x = self.norm2(self.alpha.get_alpha() * x + self.ffn(x))
        self.last_activations = x
        return x
    
    def get_weights(self):
        weights = self.attn.weight_matrices()
        weights["W1"] = self.ffn[0].weight.detach().cpu().numpy()
        weights["W2"] = self.ffn[3].weight.detach().cpu().numpy()
        return weights

class ConfigurableTransformer(nn.Module):
    """
    Configurable transformer for different architectures:
    - Tiny: 2 layers, 64d, 4 heads
    - Small: 4 layers, 128d, 8 heads
    - Medium: 6 layers, 256d, 8 heads
    - Large: 8 layers, 512d, 16 heads
    """
    def __init__(self, vocab_size=16, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, seq_len=32, use_pos=True, causal=False,
                 dropout=0.1, task_type='seq2seq', num_classes=10,
                 add_noise=NoiseScheduler(), alpha=ResAlphaScheduler()):
        super().__init__()
        self.seq_len = seq_len
        self.task_type = task_type
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        self.embed = nn.Embedding(vocab_size, d_model)
        self.use_pos = use_pos
        if use_pos:
            self.pos_embed = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, nhead, dim_feedforward, dropout, causal, add_noise, alpha)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Task-specific heads
        if task_type in ['seq2seq', 'causal_lm']:
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == 'classification':
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )

    def forward(self, x):
        B, T = x.shape
        x = x.to(device)
        emb = self.embed(x)
        if self.use_pos:
            pe = self.pos_embed[:, :T, :]
            emb = emb + pe

        out = emb
        for layer in self.layers:
            out = layer(out)

        out = self.norm(out)

        if self.task_type in ['seq2seq', 'causal_lm']:
            logits = self.head(out)
        elif self.task_type == 'classification':
            pooled = out.mean(dim=1)
            logits = self.head(pooled)

        return logits

    def get_cached_attentions(self):
        return [layer.last_attn_weights for layer in self.layers]

    def get_cached_activations(self):
        return [layer.last_activations for layer in self.layers]
    
    def get_cached_detached_activations(self):
        return [layer.last_activations.detach().cpu().numpy() for layer in self.layers]

    def get_cached_weights(self):
        return [layer.get_weights() for layer in self.layers]

# Architecture presets
ARCHITECTURES = {
    'tiny': {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 256},
    'small': {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dim_feedforward': 512},
    'medium': {'d_model': 256, 'nhead': 8, 'num_layers': 6, 'dim_feedforward': 1024},
    'large': {'d_model': 512, 'nhead': 16, 'num_layers': 8, 'dim_feedforward': 2048},
}

print("✅ Transformer architectures defined:")
for name, config in ARCHITECTURES.items():
    params = config['d_model'] * config['num_layers'] * config['nhead']
    print(f"   - {name.upper()}: {config['num_layers']} layers, {config['d_model']}d, {config['nhead']} heads (~{params//1000}K params)")

# =============================================================================
# RANK METRICS & ANALYSIS
# =============================================================================

def effective_rank(svals):
    """
    Computes the effective rank of the singular values
    """
    return np.sum(np.square(svals)) / np.square(np.sum(svals))

def stable_rank_from_svals(svals):
    """Stable rank: ||A||_F^2 / ||A||_2^2"""
    s = np.array(svals)
    if s.size == 0:
        return 0.0
    fro_sq = np.sum(s**2)
    op_sq = (s[0]**2) if s.size > 0 else 1e-12
    return float(max(1e-12, fro_sq / (op_sq + 1e-12)))

def entropy_effective_rank(svals):
    """Entropy-based effective rank: exp(H(p))"""
    s = np.array(svals)
    if s.size == 0:
        return 0.0
    p = s / (s.sum() + 1e-12)
    p = p[p > 0]
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(ent))

def topk_energy_fraction(svals, k=1):
    """Fraction of energy in top-k singular values"""
    s = np.array(svals)
    if s.size == 0:
        return 0.0
    energy = np.sum(s**2)
    topk = np.sum(s[:k]**2)
    return float(topk / (energy + 1e-12))

def attn_mean_metrics(layer_attn_concat, _, activations):
    per_layer_metrics = {}
    for li, attn_tensor in enumerate(layer_attn_concat):
        if attn_tensor is not None:
            m = compute_layer_attention_metrics_mean(attn_tensor)
        else:
            m = {'error': 'no attention captured'}
        per_layer_metrics[f'layer{li}'] = m
    return per_layer_metrics

def compute_layer_attention_metrics_mean(attn_tensor):
    """
    Compute comprehensive attention rank metrics over the
    mean of the heads (easy way to see the general trend;
    not completely scientifically accurate)

    Args:
        attn_tensor: (B, nhead, T, T) attention weights

    Returns:
        dict of metrics
    """
    if attn_tensor is None:
        return {}

    B, nhead, T, _ = attn_tensor.shape
    mean_per_head = attn_tensor.mean(axis=0)  # (nhead, T, T)
    mean_across_heads = mean_per_head.mean(axis=0)  # (T, T)

    metrics = {}

    # Per-head metrics
    head_stable = []
    head_entropy_rank = []
    head_top1 = []

    for h in range(nhead):
        M = mean_per_head[h]
        try:
            svals = np.linalg.svdvals(M)
        except:
            svals = np.linalg.svdvals(M + 1e-12 * np.eye(T))

        head_stable.append(stable_rank_from_svals(svals))
        head_entropy_rank.append(entropy_effective_rank(svals))
        head_top1.append(topk_energy_fraction(svals, k=1))

    metrics['head_stable_mean'] = float(np.mean(head_stable))
    metrics['head_stable_std'] = float(np.std(head_stable))
    metrics['head_stable_min'] = float(np.min(head_stable))
    metrics['head_stable_max'] = float(np.max(head_stable))
    metrics['head_entropy_mean'] = float(np.mean(head_entropy_rank))
    metrics['head_top1_mean'] = float(np.mean(head_top1))

    # Aggregated attention metrics
    M = mean_across_heads
    try:
        svals = np.linalg.svdvals(M)
    except:
        svals = np.linalg.svdvals(M + 1e-12 * torch.eye(T))

    metrics['mean_stable'] = stable_rank_from_svals(svals)
    metrics['mean_entropy_rank'] = entropy_effective_rank(svals)
    metrics['mean_top1'] = topk_energy_fraction(svals, k=1)
    metrics['mean_top3'] = topk_energy_fraction(svals, k=3)
    metrics['mean_top5'] = topk_energy_fraction(svals, k=5)

    # Attention sparsity
    attn_flat = attn_tensor.flatten()
    metrics['sparsity'] = float(np.mean(attn_flat < 0.01))
    metrics['mean_attn'] = float(attn_flat.mean())
    metrics['std_attn'] = float(attn_flat.std())

    return metrics

def attention_head_covariance(attn_head):
    """
    attn_head: (B, T, T) for a single head
    Flatten distributions and compute covariance PCA.
    """
    B, T, _ = attn_head.shape
    
    # Each row is a distribution over T tokens
    X = attn_head.reshape(B * T, T)
    
    # Mean center
    X_centered = X - X.mean(dim=0, keepdim=True)
    
    # Covariance
    cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
    cov = cov.detach().cpu().numpy()
    
    # Eigen-decomposition
    evals, evecs = np.linalg.eigh(cov)
    
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    return evals, evecs

def analyze_layer_attention(attn_tensor, layer_idx, save_prefix=None):
    """
    attn_tensor: (B, H, T, T)
    Computes covariance PCA per head and makes plots.
    """
    B, H, T, _ = attn_tensor.shape

    all_spectra = []

    for h in range(H):
        S, U = attention_head_covariance(attn_tensor[:, h])

        all_spectra.append(S)

        # === Plot eigenvalue spectrum decay ===
        plt.figure(figsize=(6,4))
        plt.plot(S.cpu().numpy())
        plt.title(f"Layer {layer_idx} Head {h} — Spectrum Decay")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue magnitude")
        plt.yscale("log")
        plt.grid(True)
        if save_prefix:
            plt.savefig(f"{save_prefix}_layer{layer_idx}_head{h}_spectrum.png",
                        dpi=200, bbox_inches="tight")
        plt.close()

        # === PCA scatter of the first 2 PCs ===
        # Project all attention vectors onto first 2 PCs
        X = attn_tensor[:, h].reshape(B * T, T)
        Xc = X - X.mean(dim=0, keepdim=True)
        pcs = Xc @ U[:, :2]    # shape (B*T, 2)

        plt.figure(figsize=(6,4))
        plt.scatter(pcs[:,0].cpu(), pcs[:,1].cpu(), s=3, alpha=0.3)
        plt.title(f"Layer {layer_idx} Head {h} — PCA (first 2 components)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        if save_prefix:
            plt.savefig(f"{save_prefix}_layer{layer_idx}_head{h}_pca.png",
                        dpi=200, bbox_inches="tight")
        plt.close()

    return all_spectra

def compute_svals(M):
    """Compute singular values robustly."""
    try:
        return np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        # small noise to stabilize
        M = M + 1e-6 * np.random.randn(*M.shape)
        return np.linalg.svd(M, compute_uv=False)

def stable_rank_from_svals(s):
    """Stable rank: ||W||_F^2 / ||W||_2^2."""
    fro = np.sum(s * s)
    top = np.max(s)
    return fro / (top * top + 1e-12)

def spectral_decay_slope(s):
    """
    Fit log-log slope of singular value spectrum.
    Large magnitude (more negative) -> faster decay -> more collapsed spectrum.
    """
    s = s[s > 1e-8]     # avoid zeros
    if len(s) < 3:
        return np.nan
    x = np.log(np.arange(1, len(s)+1))
    y = np.log(s)
    slope, _ = np.polyfit(x, y, 1)
    return slope

def attention_head_cov_spectrum(att):
    """
    att: (B, H, T, T)
    Return eigenvalues of covariance of flattened attention distributions.
    """
    B, H, T, _ = att.shape
    X = att.reshape(B * H * T, T)
    X = X - X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / (X.shape[0] - 1)
    evals, _ = np.linalg.eigh(cov)
    return np.sort(evals)[::-1]

def spectral_rank_metrics(layer_attn_concat, layer_weights_concat):
    """
    Returns:
        per_layer[layer_index] = {
            "Wq": {...}, "Wk": {...}, ..., "attn": {...}
        }
    """
    num_layers = len(layer_attn_concat)
    results = []

    for li in range(num_layers):
        layer_dict = {}

        # === Attention covariance PCA ===
        att = layer_attn_concat[li]
        if att is not None:
            evals = attention_head_cov_spectrum(att)
            layer_dict["attn"] = {
                "cov_evals": evals,
                "cov_stable_rank": float(
                    np.sum(evals) / (np.max(evals) + 1e-12)
                ),
            }
        else:
            layer_dict["attn"] = None

        for key in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]:
            mats = layer_weights_concat.get(key, None)
            if mats is None or mats[li] is None:
                layer_dict[key] = None
                continue

            M = mats[li]
            svals = compute_svals(M)
            sr = stable_rank_from_svals(svals)
            slope = spectral_decay_slope(svals)

            layer_dict[key] = {
                "svals": svals,
                "rank": int((svals > 1e-6).sum()),
                "stable_rank": float(sr),
                "spectral_slope": float(slope),
                "top_sv": float(svals[0]),
                "fro_norm": float(np.linalg.norm(svals)),
            }

        results.append(layer_dict)

    return results

def full_spectrum_metrics(layer_attn_concat, layer_weights_concat, layer_activations):
    per_layer = {}

    num_layers = len(layer_attn_concat)

    for li in range(num_layers):
        layer_dict = {}

        attn_tensor = layer_attn_concat[li]   # (B, H, T, T) or None

        if attn_tensor is not None:
            layer_dict["attn_mean_metrics"] = compute_layer_attention_metrics_mean(attn_tensor)

            # Covariance PCA per head
            B, H, T, _ = attn_tensor.shape
            cov_eigs_per_head = []
            for h in range(H):
                # shape (B, T, T)
                head = attn_tensor[:, h]
                # flatten: (B*T, T)
                X = head.reshape(B*T, T)
                X = X - X.mean(axis=0, keepdims=True)
                cov = (X.T @ X) / (X.shape[0] - 1 + 1e-8)
                evals, _ = np.linalg.eigh(cov)
                evals = np.sort(evals)[::-1]
                cov_eigs_per_head.append(evals)

            layer_dict["attn_cov_eigs_per_head"] = cov_eigs_per_head

        else:
            layer_dict["attn_mean_metrics"] = {}
            layer_dict["attn_cov_eigs_per_head"] = None

        layer_dict["weights"] = {}
        for key in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]:
            W = layer_weights_concat[key][li]  # weight is repeated; take first

            if W is None:
                layer_dict["weights"][key] = None
                continue

            M = W
            try:
                svals = np.linalg.svd(M, compute_uv=False)
            except:
                M = M + 1e-6 * np.random.randn(*M.shape)
                svals = np.linalg.svd(M, compute_uv=False)

            layer_dict["weights"][key] = {
                "svals": svals,
                "rank": int((svals > 1e-6).sum()),
                "stable_rank": float(stable_rank_from_svals(svals)),
                "top_sv": float(svals[0]),
                "fro_norm": float(np.linalg.norm(svals))
            }

        H_act = layer_activations[li]  # (B, T, d) torch tensor or None
        if H_act is not None:
            H_np = H_act
            B, T, d = H_np.shape
            X = H_np.reshape(B*T, d)
            Xc = X - X.mean(axis=0, keepdims=True)

            try:
                svals_act = np.linalg.svd(Xc, compute_uv=False)
            except:
                Xc = Xc + 1e-6 * np.random.randn(*Xc.shape)
                svals_act = np.linalg.svd(Xc, compute_uv=False)

            layer_dict["activation_svals"] = svals_act
            layer_dict["activation_stable_rank"] = float(stable_rank_from_svals(svals_act))
            layer_dict["activation_top_sv"] = float(svals_act[0])

            cov = (Xc.T @ Xc) / (Xc.shape[0] - 1 + 1e-8)
            evals, _ = np.linalg.eigh(cov)
            evals = np.sort(evals)[::-1]
            total = np.sum(evals) + 1e-12
            explained = evals[:3] / total
            layer_dict["activation_pca_explained"] = explained

            norms = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
            Xn = Xc / norms
            idx = np.random.choice(Xn.shape[0], size=min(2000, Xn.shape[0]), replace=False)
            sample = Xn[idx]
            sims = sample @ sample.T
            upper = sims[np.triu_indices_from(sims, k=1)]
            layer_dict["token_cosine_mean"] = float(upper.mean())
            layer_dict["token_cosine_std"] = float(upper.std())
            layer_dict["activation_variance"] = float(Xc.var())
        else:
            layer_dict["activation_svals"] = None
            layer_dict["activation_stable_rank"] = None
            layer_dict["activation_top_sv"] = None
            layer_dict["activation_pca_explained"] = None
            layer_dict["token_cosine_mean"] = None
            layer_dict["token_cosine_std"] = None
            layer_dict["activation_variance"] = None

        per_layer[f"layer{li}"] = layer_dict

    return per_layer

print("✅ Rank metrics defined:")
print("   - Stable rank (Frobenius / Operator norm)")
print("   - Entropy effective rank")
print("   - Top-k energy fraction")
print("   - Attention sparsity")

def create_dataset(task, seq_len, vocab_size, dataset_size, split):
    """Factory for creating datasets"""
    if task == 'copy':
        return CopyDataset(seq_len, vocab_size, dataset_size), 'seq2seq', vocab_size, 2
    elif task == 'reverse':
        return ReverseDataset(seq_len, vocab_size, dataset_size), 'seq2seq', vocab_size, 2
    elif task == 'sort':
        return SortDataset(seq_len, vocab_size, dataset_size), 'seq2seq', vocab_size, 2
    elif task == 'text':
        ds = TextPredictionDataset(seq_len, dataset_size)
        return ds, 'causal_lm', ds.vocab_size, ds.vocab_size
    elif task == 'image':
        ds = ImageClassificationDataset(8, 10, dataset_size)
        return ds, 'classification', 256, 10
    elif task == "tiny_stories":
        ds = TinyStoriesDataset(seq_len, split=split, dataset_size=dataset_size)
        return ds, 'causal_lm', ds.vocab_size, ds.vocab_size
    else:
        raise ValueError(f"Unknown task: {task}")


def run_experiment(train_dataset, probe_dataset, model,
                   arch_config, steps=2000, batch_size=64,
                   checkpoint_every=100, lr=3e-4, seed=42,
                   loss_scheduler=TrainingScheduler(),
                   metric_func=full_spectrum_metrics,
                   log_store_name="temp.json", weight_decay=0.01):
    """Run a complete training experiment with attention probing"""
    set_seed(seed)

    # Create datasets
    train_ds, task_type, _, _ = train_dataset
    probe_ds, _, _, _ = probe_dataset

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {arch_config['num_layers']} layers, {arch_config['d_model']}d, {arch_config['nhead']} heads")
    print(f"Parameters: {num_params:,}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    probe_loader = DataLoader(probe_ds, batch_size=64, shuffle=False)

    metrics_log = []
    step = 0
    start_time = time.time()

    pbar = tqdm(total=steps, desc=f"Training", unit="step")

    while step < steps:
        for xb, yb in train_loader:
            model.train()
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)

            loss_kwargs = {
                "task_type": task_type,
                "activations": model.get_cached_activations(),
                "layers": [
                    {
                        "Wq": layer.attn.q_proj.weight,
                        "Wk": layer.attn.k_proj.weight,
                    } for layer in model.layers
                ]
            }

            loss = loss_scheduler.call_function(logits, yb, **loss_kwargs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Checkpoint: Probe attention
            if step % checkpoint_every == 0 or step == 1:
                model.eval()
                pred = logits.argmax(dim=-1)
                train_correct = (pred == yb).float().sum().item()
                train_total = yb.numel()
                train_acc = train_correct / train_total
                with torch.no_grad():
                    # Collect attention from probe set
                    layer_attn_list = [[] for _ in range(arch_config['num_layers'])]
                    layer_weights_list = {
                        "Wq": [],
                        "Wk": [],
                        "Wv": [],
                        "Wo": [],
                        "W1": [],
                        "W2": []
                    }
                    for pb_x, pb_y in probe_loader:
                        pb_x = pb_x.to(device)
                        _ = model(pb_x)
                        atts = model.get_cached_attentions()
                        for li, a in enumerate(atts):
                            if a is not None:
                                layer_attn_list[li].append(a)
                    weights = model.get_cached_weights()
                    for li, weight in enumerate(weights):
                        for key in layer_weights_list:
                            if weight[key] is not None:
                                layer_weights_list[key].append(weight[key])

                    # Concatenate batches
                    layer_attn_concat = []
                    for li in range(arch_config['num_layers']):
                        if len(layer_attn_list[li]) > 0:
                            layer_attn_concat.append(np.concatenate(layer_attn_list[li], axis=0))
                        else:
                            layer_attn_concat.append(None)
                    # Compute metrics per layer
                    per_layer_metrics = metric_func(
                        layer_attn_concat, layer_weights_list,
                        model.get_cached_detached_activations())

                    # Evaluate accuracy
                    total_correct = 0
                    total_samples = 0
                    eval_loss = 0
                    for xb_eval, yb_eval in probe_loader:
                        xb_eval, yb_eval = xb_eval.to(device), yb_eval.to(device)
                        logits_eval = model(xb_eval)
                        loss_kwargs = {
                            "task_type": task_type,
                            "activations": model.get_cached_activations(),
                            "layers": [
                                {
                                    "Wq": layer.attn.q_proj.weight,
                                    "Wk": layer.attn.k_proj.weight,
                                } for layer in model.layers
                            ]
                        }
                        if task_type in ['seq2seq', 'causal_lm']:
                            B_e, T_e, V = logits_eval.shape
                            total_samples += B_e * T_e
                        else:
                            total_samples += yb_eval.shape[0]
                        eval_loss += loss_scheduler.call_function(
                            logits_eval, yb_eval, **loss_kwargs).item()
                        pred = logits_eval.argmax(dim=-1)
                        total_correct += (pred == yb_eval).float().sum().item()

                    acc = total_correct / total_samples
                    eval_loss = eval_loss / len(probe_loader)
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

                    metrics_log.append({
                        'step': step,
                        'time': time.time() - start_time,
                        'train_loss': float(loss.item()),
                        'train_acc': float(train_acc),
                        'eval_loss': float(eval_loss),
                        'eval_acc': float(acc),
                        'per_layer': per_layer_metrics
                    })

                    np.save(log_store_name, metrics_log, allow_pickle=True)

                model.train()

            if step >= steps:
                break

    pbar.close()
    print(f"✅ Training complete in {time.time()-start_time:.1f}s")
    return model, metrics_log


print("✅ Training system ready!")


# =============================================================================
# VISUALIZATION & ANALYSIS
# =============================================================================

def plot_rank_evolution(metrics_log, metric_keys=['mean_stable', 'mean_entropy_rank', 'head_stable_mean']):
    """Plot how attention rank evolves during training"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Extract steps and layers
    steps = [m['step'] for m in metrics_log]
    layer_names = list(metrics_log[0]['per_layer'].keys())
    num_layers = len(layer_names)

    # Plot each metric
    for idx, metric_key in enumerate(metric_keys[:3]):
        ax = axes[idx]
        for layer_name in layer_names:
            values = [m['per_layer'][layer_name].get(metric_key, np.nan) for m in metrics_log]
            ax.plot(steps, values, marker='o', label=layer_name, linewidth=2)

        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel(metric_key.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric_key.replace("_", " ").title()} Evolution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # Plot accuracy
    ax = axes[3]
    accs = [m['eval_acc'] for m in metrics_log]
    ax.plot(steps, accs, marker='o', color='green', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Evaluation Accuracy', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_rank_trend(metrics_log, layer_idx=0, metric='mean_stable'):
    """Analyze if rank increased or decreased during training"""
    layer_name = f'layer{layer_idx}'
    values = [m['per_layer'][layer_name].get(metric, np.nan) for m in metrics_log]
    values = np.array(values)

    if len(values) < 2:
        return "Not enough data"

    start_val = values[0]
    end_val = values[-1]
    change = end_val - start_val
    pct_change = (change / start_val) * 100 if start_val != 0 else 0

    trend = "INCREASED" if change > 0 else "DECREASED"

    print(f"\n📊 Rank Analysis for {layer_name} ({metric}):")
    print(f"   Initial: {start_val:.3f}")
    print(f"   Final: {end_val:.3f}")
    print(f"   Change: {change:+.3f} ({pct_change:+.1f}%)")
    print(f"   Trend: Rank {trend}")

    return trend


print("✅ Visualization tools ready!")

# =============================================================================
# Run All Experiments
# =============================================================================

increase_rank_regularizers = {
    "logdet": logdetregularizer,
    "erank": erank_loss,
    "orthonorm": orthogonal_loss,   # encouraging orthonormal Wq/Wk tends to increase rank
    # "residual_alpha" could be separate scheduler to change residual strength
}

decrease_rank_regularizers = {
    "spectral_norm": spectral_norm_loss,   # penalize top singular value
    "orthonorm": orthogonal_loss,
    # also allow side-effect noise or orthogonality+noise combo
    # "enable_noise": enable_noise_sideeffect,
    # "disable_noise": disable_noise_sideeffect
}


def instantiate_architecture(name, vocab_size=32, seq_len=32, task_type='seq2seq'):
    # returns model instance and arch config dict (for run_experiment)
    arch_config = ARCHITECTURES[name].copy()
    d_model = arch_config['d_model']
    nhead = arch_config['nhead']
    num_layers = arch_config['num_layers']
    model = ConfigurableTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                                    num_layers=num_layers, dim_feedforward=arch_config['dim_feedforward'],
                                    seq_len=seq_len, use_pos=True, causal=(task_type!='classification'))
    return model, arch_config


def timing_windows(steps):
    """
    Returns dict mapping keys -> (start, end) indices (inclusive)
    early: first 25%
    middle: 25%..75%
    late: last 25%
    all: full range
    """
    s25 = int(0.25 * steps)
    s75 = int(0.75 * steps)
    return {
        "early": (0, max(0, s25 - 1)),
        "middle": (max(0, s25), max(0, s75 - 1)),
        "late": (max(0, s75), max(0, steps - 1)),
        "all": (0, max(0, steps - 1))
    }

# --------------------------
# Sweep & combo runners
# --------------------------

def make_scheduler_for_regularizer(
        func, steps, when="all", which_layers="all", weight=1.0):
    """
    For a single loss function returns a TrainingScheduler instance
    with it applied in the specific window
    """
    windows = timing_windows(steps)
    start, end = windows[when]
    ts = TrainingScheduler(
        which_layers=which_layers, base_func=cross_entropy,
        functions_at_times=[
            (start, end, lambda logits, yb, **kwargs: weight * func(logits, yb, **kwargs))
        ])
    return ts

def run_single_config(task_name, dataset_size_name, arch_name, reg_name,
                      reg_func, when, weight=1.0, steps=2000, batch_size=64,
                      lr=3e-4, weight_decay=0.01, out_dir="logs", seed=42):
    """
    Build dataset, model, scheduler for the given config and run run_experiment.
    Saves logs as npy in out_dir with descriptive name.
    Returns path to saved file.
    """
    set_seed(seed)
    sizes = {"small": 1024, "medium": 4096, "large": 8192}
    dataset_size = sizes[dataset_size_name]

    # create datasets
    ds_train, task_type, vocab_size, _ = create_dataset(
        task_name, seq_len=32, vocab_size=64, dataset_size=dataset_size,
        split="train")
    ds_probe, _, _, _ = create_dataset(
        task_name, seq_len=32, vocab_size=64,
        dataset_size=min(512, dataset_size), split="train")

    # instantiate model
    model, arch_config = instantiate_architecture(
        arch_name, vocab_size=max(32, vocab_size), seq_len=32, task_type=task_type)
    model = model.to(device)

    # build scheduler
    scheduler = make_scheduler_for_regularizer(
        reg_func, steps=steps, when=when, which_layers="all", weight=weight)

    # run experiment; ensure run_experiment uses kwargs activations, layers, and model
    log_name = f"{task_name}_{dataset_size_name}_{arch_name}_{reg_name}_{when}_steps{steps}_seed{seed}.npy"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, log_name)

    model, _ = run_experiment((ds_train, task_type, None, None),
                (ds_probe, task_type, None, None),
                model, arch_config,
                steps=steps, batch_size=batch_size,
                checkpoint_every=max(1, steps // 20), lr=lr,
                seed=seed, loss_scheduler=scheduler, metric_func=full_spectrum_metrics,
                log_store_name=out_path, weight_decay=weight_decay)

    print(f"Saved logs to {out_path}")
    return out_path

def run_sweep(tasks=None, archs=None, dataset_sizes=None,
              increase_dict=None, decrease_dict=None, whens=None,
              steps=2000, batch_size=64, out_dir="logs", seed=42):
    """
    Iterate over tasks × archs × dataset sizes × regs × timings and run experiments.
    By default uses the increase_dict keys then decrease_dict keys.
    """
    if tasks is None:
        tasks = ["copy", "reverse", "sort", "text", "tiny_stories", "image"]
    if archs is None:
        archs = ["tiny", "small", "medium"]
    if dataset_sizes is None:
        dataset_sizes = ["small", "medium", "large"]
    if increase_dict is None:
        increase_dict = increase_rank_regularizers
    if decrease_dict is None:
        decrease_dict = decrease_rank_regularizers
    if whens is None:
        whens = ["early", "middle", "late", "all"]

    all_runs = []

    for task in tasks:
        for size_name in dataset_sizes:
            for arch in archs:
                # first do increase-rank experiments
                for reg_name, reg_func in increase_dict.items():
                    for when in whens:
                        print(f"RUN: task={task} size={size_name} arch={arch} reg={reg_name} when={when}")
                        run_single_config(task, size_name, arch, reg_name, reg_func, when,
                                          steps=steps, batch_size=batch_size, out_dir=out_dir, seed=seed)
                # then decrease-rank experiments
                for reg_name, reg_func in decrease_dict.items():
                    for when in whens:
                        print(f"RUN: task={task} size={size_name} arch={arch} reg={reg_name} when={when}")
                        run_single_config(task, size_name, arch, reg_name, reg_func, when,
                                          steps=steps, batch_size=batch_size, out_dir=out_dir, seed=seed)

    return all_runs

# --------------------------
# Collapse <-> Expansion combos
# --------------------------

def run_collapse_expand_combos(
        task_name="copy", dataset_size_name="medium", arch="small",
        collapse_func=None, expand_func=None, steps=2000,
        batch_size=64, out_dir="logs", seed=42):
    """
    Runs:
      - collapse early (first25) then expand late (last25)
      - expand early then collapse late
      - collapse early only
      - expand early only
    Uses collapse_func and expand_func from dictionaries.
    """
    combos = [
        ("collapseEarlyThenExpandLate", [ (0, int(0.25*steps)-1, collapse_func), (int(0.75*steps), steps-1, expand_func) ]),
        ("expandEarlyThenCollapseLate", [ (0, int(0.25*steps)-1, expand_func), (int(0.75*steps), steps-1, collapse_func) ]),
        ("collapseEarlyOnly", [ (0, int(0.25*steps)-1, collapse_func) ]),
        ("expandEarlyOnly", [ (0, int(0.25*steps)-1, expand_func) ]),
    ]
    sizes = {"small": 1024, "medium": 4096, "large": 8192}

    for combo_name, funcs in combos:
        print(f"RUN combo {combo_name}")
        # build TrainingScheduler from funcs mapping to (start,end,func)
        scheduler = TrainingScheduler(which_layers="all", base_func=cross_entropy, functions_at_times=funcs)
        # build datasets & model
        dataset_size = sizes[dataset_size_name]
        ds_train, task_type, vocab_size, _ = create_dataset(
            task_name, seq_len=32, vocab_size=64, dataset_size=dataset_size,
            split="train")
        ds_probe, _, _, _ = create_dataset(
            task_name, seq_len=32, vocab_size=64,
            dataset_size=min(512, dataset_size), split="train")
        model, arch_config = instantiate_architecture(arch, vocab_size=max(32, vocab_size), seq_len=32, task_type=task_type)
        model = model.to(device)

        # wrapper loss to ensure scheduled functions get model/layers/activations
        # we reuse run_experiment but pass scheduler object directly: run_experiment calls scheduler.call_function with **kwargs
        log_name = f"{task_name}_{dataset_size_name}_{arch}_{combo_name}_steps{steps}_seed{seed}.npy"
        out_path = os.path.join(out_dir, log_name)
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(out_path):
            model, _ = run_experiment((ds_train, task_type, None, None),
                (ds_probe, task_type, None, None),
                model, arch_config,
                steps=steps, batch_size=batch_size,
                checkpoint_every=max(1, steps//10), lr=3e-4,
                seed=seed, loss_scheduler=scheduler, metric_func=full_spectrum_metrics,
                log_store_name=out_path, weight_decay=0.01)
            print(f"Saved combo log to {out_path}")
        else:
            print(f"{out_path} already exists in the folder")

def run_collapse_expand_sweeps(
        collapse_func, expand_func, tasks=None, archs=None,
        dataset_sizes=None, steps=2000, batch_size=64,
        out_dir="logs", seed=42):
    """
    Iterate over tasks × archs × dataset sizes × regs × timings and run experiments.
    By default uses the increase_dict keys then decrease_dict keys.
    """
    if tasks is None:
        tasks = ["copy", "reverse", "sort", "text", "tiny_stories", "image"]
    if archs is None:
        archs = ["tiny", "small", "medium"]
    if dataset_sizes is None:
        dataset_sizes = ["small", "medium", "large"]

    all_runs = []

    for task in tasks:
        for size_name in dataset_sizes:
            for arch in archs:
                print(f"RUN: task={task} size={size_name} arch={arch}")
                run_collapse_expand_combos(
                    task, size_name, arch, collapse_func, expand_func, steps,
                    batch_size, out_dir=out_dir, seed=seed)
    return all_runs


# --------------------------
# Plotting helpers
# --------------------------

def plot_activation_spectrum(H, layer_idx=None, save_prefix=None):
    """
    H: activation tensor shape (B, T, d)
    Plot eigen-spectrum of activation covariance (per-layer).
    """
    H = H.detach().cpu()
    B, T, d = H.shape
    X = H.reshape(B * T, d).numpy()
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]
    plt.figure()
    plt.semilogy(evals, marker='o')
    title = f"Activation spectrum"
    if layer_idx is not None:
        title += f" layer {layer_idx}"
    plt.title(title)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue (log scale)")
    if save_prefix:
        plt.savefig(f"{save_prefix}_actspec_layer{layer_idx}.png", dpi=200)
    plt.close()
    return evals

def plot_erank_over_time(metrics_log, layer_index=0, save_path=None):
    """
    Given metrics_log returned by run_experiment (list of dicts),
    plot per-checkpoint effective rank (we use mean_stable as proxy if available).
    """
    steps = [m['step'] for m in metrics_log]
    vals = []
    for m in metrics_log:
        per_layer = m.get('per_layer', {})
        layer_key = f'layer{layer_index}'
        if layer_key in per_layer:
            v = per_layer[layer_key].get('mean_stable', np.nan)
        else:
            v = np.nan
        vals.append(v)
    plt.figure()
    plt.plot(steps, vals, marker='o')
    plt.xlabel("Training step")
    plt.ylabel("Stable rank (mean)")
    plt.title(f"Layer {layer_index} stable-rank over training")
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()