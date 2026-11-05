import torch
from torch import nn

class SanskritEncoder(nn.Module):
    """Compact Transformer encoder with MLM and token-level morphology heads."""
    def __init__(self, vocab_size, num_tags, hidden=256, layers=4, heads=4, max_len=128, dropout=.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.position = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden*4, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.mlm = nn.Linear(hidden, vocab_size)
        self.morph = nn.Linear(hidden, num_tags)
        self.max_len = max_len

    def forward(self, ids, attention_mask=None):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.token(ids) + self.position(pos)
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        h = self.norm(self.encoder(x, src_key_padding_mask=pad_mask))
        return {"hidden": h, "mlm_logits": self.mlm(h), "morph_logits": self.morph(h)}
