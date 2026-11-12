"""
Confidence-filtered pseudo-label generation + guarded continual update.
"""
import argparse, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils.vocab import SanskritTokenizer
from grammar.paninian_engine import consistency_score


TAGS = ["PAD","UNK","NOUN","PROPN","VERB","ADJ","ADV","PRON","DET","ADP","CCONJ","SCONJ","NUM","PART","PUNCT","X"]
tag2id = {t: i for i, t in enumerate(TAGS)}


def _parse_conllu(path: Path):
    """Parse CoNLL-U → list of (forms, tags)."""
    sentences, forms, upos = [], [], []
    if not path.exists():
        return sentences
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if not line:
            if forms:
                sentences.append((list(forms), list(upos)))
                forms.clear(); upos.clear()
            continue
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) == 10 and "-" not in c[0] and "." not in c[0]:
            forms.append(c[1])
            upos.append(c[3] if c[3] in tag2id else "X")
    return sentences


def _eval_morph_f1(model, tok, device, sentences):
    """Compute macro-F1 on a list of (forms, tags) using sklearn."""
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        return None
    all_pred, all_true = [], []
    model.eval()
    with torch.no_grad():
        for forms, tags in sentences:
            text = " ".join(forms)
            ids_list = tok.encode(text, max_length=128)
            if not ids_list:
                continue
            ids   = torch.tensor([ids_list], device=device)
            mask  = torch.ones_like(ids)
            out   = model(ids, mask)
            preds = out["morph_logits"][0].argmax(-1).cpu().tolist()
            n     = min(len(preds), len(tags))
            all_pred.extend(preds[:n])
            all_true.extend([tag2id.get(t, 1) for t in tags[:n]])
    if not all_true:
        return None
    return float(f1_score(all_true, all_pred, average="macro", zero_division=0))


def run(max_pseudo: int = 2000):
    ck        = ROOT / "checkpoints"
    cp        = ck / "current.pt"
    vocab_path = ck / "vocab.json"

    if not cp.exists() or not vocab_path.exists():
        raise SystemExit("Train first: python train.py --epochs 1 --smoke")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state  = torch.load(cp, map_location=device, weights_only=False)
    tok    = SanskritTokenizer(str(vocab_path))
    tags   = state.get("tags", TAGS)

    from models.network import SanskritEncoder
    model = SanskritEncoder(state["vocab_size"], len(tags)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # ---- Pseudo-label generation ----
    source = ROOT / "data" / "processed" / "mlm_train.txt"
    lines  = source.read_text(encoding="utf-8").splitlines() if source.exists() else []

    pseudo, review = [], []
    for line in lines[:max_pseudo]:
        line = line.strip()
        if not line:
            continue
        toks_str = tok.tokenize(line)
        ids_list = tok.encode(line, max_length=128)
        if not ids_list:
            continue
        ids  = torch.tensor([ids_list], device=device)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out   = model(ids, mask)
            probs = out["morph_logits"].softmax(-1).max(-1).values[0]
            conf  = float(probs.mean().item())
        gram_score, _ = consistency_score(toks_str)
        entry = {"text": line, "confidence": conf, "grammar_score": gram_score}
        if conf >= 0.72 and gram_score >= 0.65:
            pseudo.append(entry)
        elif conf < 0.72:
            review.append({**entry, "reason": "uncertain prediction"})

    (ck / "pseudo_labels.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pseudo),
        encoding="utf-8",
    )
    (ck / "active_learning_queue.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in review[:500]),
        encoding="utf-8",
    )

    # ---- Guarded evaluation ----
    conllu   = ROOT / "data" / "processed" / "sa_vedic_train.conllu"
    sents    = _parse_conllu(conllu)
    # Use last 10% as validation hold-out
    n_val    = max(1, len(sents) // 10)
    val_sents = sents[-n_val:]

    current_f1 = _eval_morph_f1(model, tok, device, val_sents)

    baseline_path = ck / "baseline_morph_f1.json"
    if baseline_path.exists():
        baseline_f1 = json.loads(baseline_path.read_text(encoding="utf-8")).get("f1")
    else:
        baseline_f1 = None
        if current_f1 is not None:
            baseline_path.write_text(json.dumps({"f1": current_f1}), encoding="utf-8")

    promoted = False
    promo_reason = "No independent validated evaluation improvement demonstrated."
    if current_f1 is not None and baseline_f1 is not None:
        if current_f1 - baseline_f1 >= 0.001:
            promoted = True
            promo_reason = f"Improvement {current_f1:.4f} vs baseline {baseline_f1:.4f} >= 0.001"
            baseline_path.write_text(json.dumps({"f1": current_f1}), encoding="utf-8")

    report = {
        "pseudo_candidates":   len(pseudo),
        "human_review_queue":  min(500, len(review)),
        "current_morph_f1":    current_f1,
        "baseline_morph_f1":   baseline_f1,
        "promoted":            promoted,
        "reason":              promo_reason,
        "status": (
            "pseudo-label buffer generated; checkpoint promoted" if promoted
            else "pseudo-label buffer generated; checkpoint unchanged"
        ),
    }
    (ck / "self_learning_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-pseudo", type=int, default=2000)
    a = p.parse_args()
    run(a.max_pseudo)
