"""
SanskritEncoder: compact Transformer encoder.
- Weight-tied MLM head
- Token-level morphology/POS head
- Sentence pooling (CLS + mean) for future downstream tasks
- Config-driven from configs/model.yaml
"""
import yaml
from pathlib import Path
import torch
from torch import nn

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "model.yaml"


def _load_model_cfg():
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


class SanskritEncoder(nn.Module):
    def __init__(self, vocab_size, num_tags,
                 hidden=None, layers=None, heads=None,
                 max_len=None, dropout=None):
        super().__init__()
        cfg     = _load_model_cfg()
        hidden  = hidden  or cfg.get("hidden_size", 256)
        layers  = layers  or cfg.get("num_layers",  4)
        heads   = heads   or cfg.get("num_heads",   4)
        max_len = max_len or cfg.get("max_length",  128)
        dropout = dropout if dropout is not None else cfg.get("dropout", 0.1)

        self.hidden   = hidden
        self.max_len  = max_len
        self.num_tags = num_tags

        # Embeddings
        self.token    = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.position = nn.Embedding(max_len, hidden)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout,
            batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)
        self.norm    = nn.LayerNorm(hidden)

        # Weight-tied MLM head
        self.mlm_proj = nn.Linear(hidden, hidden)
        self.mlm_act  = nn.GELU()
        self.mlm_norm = nn.LayerNorm(hidden)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

        # Morphology / POS head
        self.morph = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_tags),
        )

        # Sentence pooler: CLS + mean-pool → dense
        self.pooler = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    # ------------------------------------------------------------------
    def _encode(self, ids, attention_mask=None):
        b, t = ids.shape
        t = min(t, self.max_len)
        ids = ids[:, :t]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :t]
        pos      = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x        = self.token(ids) + self.position(pos)
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        h        = self.norm(self.encoder(x, src_key_padding_mask=pad_mask))
        return h, attention_mask

    def _pool(self, h, attention_mask=None):
        cls  = h[:, 0]
        if attention_mask is not None:
            m    = attention_mask.float().unsqueeze(-1)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            mean = h.mean(1)
        return self.pooler(torch.cat([cls, mean], dim=-1))

    # ------------------------------------------------------------------
    def forward(self, ids, attention_mask=None):
        h, attention_mask = self._encode(ids, attention_mask)

        # MLM: proj → act → norm → weight-tied linear + bias
        mlm_h   = self.mlm_norm(self.mlm_act(self.mlm_proj(h)))
        mlm_log = mlm_h @ self.token.weight.T + self.mlm_bias

        morph_log = self.morph(h)
        sent_vec  = self._pool(h, attention_mask)

        return {
            "hidden":       h,
            "mlm_logits":   mlm_log,
            "morph_logits": morph_log,
            "sentence_vec": sent_vec,
        }

    # ------------------------------------------------------------------
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
