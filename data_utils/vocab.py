import json, re
from pathlib import Path
from collections import Counter

PAD, UNK, MASK = "[PAD]", "[UNK]", "[MASK]"
def build_vocab(lines, path, limit=12000):
    counts=Counter()
    for line in lines:
        counts.update(re.findall(r"[\u0900-\u097F]+|[^\s]", line))
    tokens=[PAD,UNK,MASK,"[CLS]"]+[t for t,_ in counts.most_common(limit-4)]
    vocab={t:i for i,t in enumerate(tokens)}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(vocab,ensure_ascii=False,indent=2),encoding="utf-8")
    return vocab

class SanskritTokenizer:
    def __init__(self, path):
        self.vocab=json.loads(Path(path).read_text(encoding="utf-8"))
        self.inv={v:k for k,v in self.vocab.items()}
    def tokenize(self, text):
        return re.findall(r"[\u0900-\u097F]+|[^\s]", text)
    def encode(self, text, max_length=128):
        toks=self.tokenize(text)[:max_length]
        return [self.vocab.get(t,1) for t in toks]
    def decode(self, ids):
        return " ".join(self.inv.get(int(i),UNK) for i in ids if int(i) not in (0,2,3))
