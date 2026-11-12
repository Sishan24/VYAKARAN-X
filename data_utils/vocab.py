"""Sanskrit-aware vocabulary and tokenizer."""
import json, re, unicodedata
from pathlib import Path
from collections import Counter
import torch

PAD, UNK, MASK, CLS = "[PAD]", "[UNK]", "[MASK]", "[CLS]"
PAD_ID, UNK_ID, MASK_ID, CLS_ID = 0, 1, 2, 3

# Devanagari range + dandas + digits
_DEVANAGARI = re.compile(r"[\u0900-\u097F\u0964\u0965]+|[^\s]")


def _tokenize_raw(text: str):
    """Split on Devanagari tokens (including dandas) and non-whitespace singles."""
    text = unicodedata.normalize("NFC", text)
    tokens = []
    for m in _DEVANAGARI.finditer(text):
        tok = m.group()
        # Split dandas out as individual tokens
        parts = re.split(r"([\u0964\u0965])", tok)
        for p in parts:
            if p:
                tokens.append(p)
    return tokens


def build_vocab(lines, path, limit=12000):
    """Build a frequency-based vocabulary from text lines."""
    counts = Counter()
    for line in lines:
        counts.update(_tokenize_raw(str(line)))
    # Reserve first 4 slots for special tokens
    special = [PAD, UNK, MASK, CLS]
    vocab_tokens = special + [t for t, _ in counts.most_common(limit - len(special))]
    vocab = {t: i for i, t in enumerate(vocab_tokens)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    return vocab


class SanskritTokenizer:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab = data
        self.inv   = {v: k for k, v in data.items()}
        self._path = path

    # ----- core -----
    def tokenize(self, text: str):
        return _tokenize_raw(str(text))

    def encode(self, text, max_length=128):
        toks = self.tokenize(text)[:max_length]
        return [self.vocab.get(t, UNK_ID) for t in toks]

    def decode(self, ids):
        skip = {PAD_ID, MASK_ID, CLS_ID}
        return " ".join(
            self.inv.get(int(i), UNK)
            for i in ids
            if int(i) not in skip
        )

    # ----- batch helpers -----
    def batch_encode(self, texts, max_length=128):
        """Return (input_ids LongTensor, attention_mask LongTensor) both [B, max_length]."""
        encoded = [self.encode(t, max_length) for t in texts]
        lens    = [len(e) for e in encoded]
        L       = min(max(lens), max_length) if lens else 1
        ids     = torch.zeros(len(texts), L, dtype=torch.long)
        mask    = torch.zeros(len(texts), L, dtype=torch.long)
        for i, e in enumerate(encoded):
            n = len(e)
            ids[i, :n]  = torch.tensor(e, dtype=torch.long)
            mask[i, :n] = 1
        return ids, mask

    # ----- properties -----
    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return self.vocab_size
