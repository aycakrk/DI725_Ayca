import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 2000
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True

class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Şekillendirme ve transpose işlemleri
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Gelecekteki tokenları maskeler
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.sentiment_head = nn.Linear(config.n_embd, 3, bias=True)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Ağırlıkların başlatılması
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay_params = []
        no_decay_params = []
        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in torch.optim.AdamW.__init__.__annotations__
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def forward(self, idx, sentiment_labels=None, targets=None):
        device = idx.device
        B, T = idx.size()

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer['wte'](idx)
        pos_emb = self.transformer['wpe'](pos)
        x = tok_emb + pos_emb
        x = self.transformer['drop'](x)

        for block in self.transformer['h']:
            x = block(x)

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)

        x_avg = x.mean(dim=1)
        sentiment_logits = self.sentiment_head(x_avg)

        loss = None
        loss_lm = None
        loss_sent = None

        if targets is not None:
            loss_lm = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        if sentiment_labels is not None:
            # Weighted loss için ağırlıklar: [1.0, 1.0, 3.0] örneğin,
            # negatif ve nötr için 1, pozitif için 3.0 ağırlık veriliyor.
            sentiment_weights = torch.tensor([1.0, 1.0, 10.0], device=device)
            loss_sent = nn.functional.cross_entropy(sentiment_logits, sentiment_labels, weight=sentiment_weights)

        if loss_lm is not None and loss_sent is not None:
            loss = loss_lm + loss_sent
        elif loss_lm is not None:
            loss = loss_lm
        elif loss_sent is not None:
            loss = loss_sent

        return logits, sentiment_logits, loss
