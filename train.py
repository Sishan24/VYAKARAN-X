"""
VYAKARANA-X: End-to-End Training & Evaluation Pipeline for Sanskrit Language Learning.
Implements:
1. Majority & Unigram-majority baselines on UD Vedic test split.
2. No-pretraining baseline: Randomly initialized Transformer trained solely on UD Vedic morphology.
3. Stage 1 (Self-Supervised Pretraining): Masked Language Modeling (MLM) on Sansar Sanskrit corpus.
4. Stage 2 (Supervised Fine-Tuning): Morphology head fine-tuning initialized from MLM weights.
5. Rigorous evaluation on held-out official UD Vedic test split (Accuracy, Macro-F1).
6. Metrics compilation into metrics/results.json and checkpoints/training_metrics.json.
"""
import argparse, json, logging, math, random, sys, time
from pathlib import Path
from collections import Counter, defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.network import SanskritEncoder
from data_utils.vocab import build_vocab, SanskritTokenizer, PAD_ID, MASK_ID, UNK_ID, CLS_ID

# Setup clean unbuffered logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vyakarana.train")

TAGS = ["PAD", "UNK", "NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "NUM", "PART", "PUNCT", "X"]
tag2id = {t: i for i, t in enumerate(TAGS)}
id2tag = {i: t for i, t in enumerate(TAGS)}


# ---------------------------------------------------------------------------
# Pure-Python Fast Metrics Calculation (No DLL or sklearn overhead)
# ---------------------------------------------------------------------------
def compute_classification_metrics(y_true, y_pred):
    """Computes accuracy, macro-F1, and weighted-F1 with zero external dependencies."""
    if not y_true or not y_pred:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = correct / max(1, total)

    classes = sorted(list(set(y_true) | set(y_pred)))
    f1s = []
    weights = []

    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        support = sum(1 for yt in y_true if yt == c)
        weights.append(support)

    macro_f1 = sum(f1s) / max(1, len(f1s))
    total_weight = sum(weights)
    weighted_f1 = sum(f * w for f, w in zip(f1s, weights)) / max(1, total_weight)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
    }


# ---------------------------------------------------------------------------
# Dataset Parsers and Torch Datasets
# ---------------------------------------------------------------------------
def parse_conllu(path: Path):
    """Parse CoNLL-U format to list of (form_tokens, upos_tags)."""
    sentences, forms, upos = [], [], []
    if not path.exists():
        log.warning(f"CoNLL-U file not found: {path}")
        return sentences
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                if forms:
                    sentences.append((list(forms), list(upos)))
                    forms.clear()
                    upos.clear()
                continue
            if line.startswith("#"):
                continue
            c = line.split("\t")
            if len(c) == 10 and "-" not in c[0] and "." not in c[0]:
                forms.append(c[1])
                upos.append(c[3] if c[3] in tag2id else "X")
        if forms:
            sentences.append((list(forms), list(upos)))
    return sentences


class MLMDataset(Dataset):
    def __init__(self, lines, tok: SanskritTokenizer, max_len: int = 128):
        self.samples = []
        for line in lines:
            ids = tok.encode(line, max_length=max_len)
            if ids:
                self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx][:128]
        pad_len = 128 - len(ids)
        return torch.tensor(ids + [PAD_ID] * pad_len, dtype=torch.long)


class MorphDataset(Dataset):
    def __init__(self, sentences, tok: SanskritTokenizer, max_len: int = 128):
        self.samples = []
        for forms, tags in sentences:
            ids = tok.encode(" ".join(forms), max_length=max_len)
            labs = [tag2id.get(t, tag2id["X"]) for t in tags][:max_len]
            n = min(len(ids), len(labs))
            if n > 0:
                self.samples.append((ids[:n], labs[:n]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labs = self.samples[idx]
        pad_len = 128 - len(ids)
        return (
            torch.tensor(ids + [PAD_ID] * pad_len, dtype=torch.long),
            torch.tensor(labs + [-100] * pad_len, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def compute_majority_baselines(train_sents, test_sents):
    """
    Computes:
    1. Majority Class Baseline: Always predicts the globally most frequent tag.
    2. Unigram Majority Baseline: For each token, predicts its most frequent training tag; fallback to global majority.
    """
    all_train_tags = []
    token_tag_counts = defaultdict(Counter)

    for forms, tags in train_sents:
        for f, t in zip(forms, tags):
            all_train_tags.append(t)
            token_tag_counts[f][t] += 1

    global_majority_tag = Counter(all_train_tags).most_common(1)[0][0]

    unigram_map = {}
    for token, counts in token_tag_counts.items():
        unigram_map[token] = counts.most_common(1)[0][0]

    y_true = []
    y_pred_majority = []
    y_pred_unigram = []

    for forms, tags in test_sents:
        for f, t in zip(forms, tags):
            y_true.append(t)
            y_pred_majority.append(global_majority_tag)
            y_pred_unigram.append(unigram_map.get(f, global_majority_tag))

    maj_metrics = compute_classification_metrics(y_true, y_pred_majority)
    uni_metrics = compute_classification_metrics(y_true, y_pred_unigram)

    return {
        "global_majority_tag": global_majority_tag,
        "majority_class": maj_metrics,
        "unigram_majority": uni_metrics,
    }


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------
def eval_morphology(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    steps = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for ids, labs in dataloader:
            ids = ids.to(device)
            labs = labs.to(device)
            out = model(ids, (ids != 0).long(), mlm=False, morph=True, pool=False)
            logits = out["morph_logits"]
            loss = loss_fn(logits.reshape(-1, len(TAGS)), labs.reshape(-1))
            total_loss += loss.item()
            steps += 1

            preds = logits.argmax(-1)
            mask = labs != -100
            all_preds.extend(preds[mask].cpu().tolist())
            all_targets.extend(labs[mask].cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_preds)
    avg_loss = total_loss / max(1, steps)
    metrics["loss"] = round(avg_loss, 4)
    return metrics


def eval_mlm(model, dataloader, device, vocab_size, mask_prob=0.15):
    model.eval()
    total_loss = 0.0
    steps = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for ids in dataloader:
            ids = ids.to(device)
            labels = ids.clone()
            mask = (torch.rand(ids.shape, device=device) < mask_prob) & (ids != 0)
            labels[~mask] = -100
            masked = ids.clone()
            masked[mask] = MASK_ID

            out = model(masked, (ids != 0).long(), mlm=True, morph=False, pool=False)
            loss = loss_fn(out["mlm_logits"].reshape(-1, vocab_size), labels.reshape(-1))
            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(1, steps)
    perplexity = math.exp(min(15.0, avg_loss))
    return round(avg_loss, 4), round(perplexity, 2)


# ---------------------------------------------------------------------------
# Training Routines
# ---------------------------------------------------------------------------
def train_morphology_head(model, train_loader, val_loader, test_loader, device, epochs, lr, name="model"):
    log.info(f"--- Training Morphology for [{name}] ({epochs} epochs, lr={lr}) ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best_val_f1 = -1.0
    best_metrics = None

    for ep in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        t0 = time.time()
        for ids, labs in train_loader:
            ids, labs = ids.to(device), labs.to(device)
            out = model(ids, (ids != 0).long(), mlm=False, morph=True, pool=False)
            loss = loss_fn(out["morph_logits"].reshape(-1, len(TAGS)), labs.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            steps += 1

        val_metrics = eval_morphology(model, val_loader, device)
        elapsed = time.time() - t0
        log.info(f"[{name}] Epoch {ep+1}/{epochs} ({elapsed:.1f}s) | Train Loss: {total_loss/max(1, steps):.4f} | Val F1: {val_metrics['macro_f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_metrics = eval_morphology(model, test_loader, device)

    return best_metrics or eval_morphology(model, test_loader, device)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VYAKARANA-X Model Training & Baselines")
    parser.add_argument("--epochs-mlm", type=int, default=2, help="Epochs for MLM pretraining")
    parser.add_argument("--epochs-morph", type=int, default=3, help="Epochs for morphology fine-tuning")
    parser.add_argument("--max-mlm-lines", type=int, default=6000, help="Max lines for MLM pretraining (bounded for CPU)")
    parser.add_argument("--max-morph-sents", type=int, default=5000, help="Max sentences from UD Vedic for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test configuration")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    device = "cpu"
    torch.set_num_threads(max(1, min(10, torch.get_num_threads())))
    log.info(f"Device: {device} | Torch Threads: {torch.get_num_threads()}")

    if args.smoke:
        args.epochs_mlm = 1
        args.epochs_morph = 1
        args.max_mlm_lines = 800
        args.max_morph_sents = 800
        args.batch_size = 8

    ck_dir = ROOT / "checkpoints"
    ck_dir.mkdir(exist_ok=True)
    metrics_dir = ROOT / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # 1. Load UD Vedic official splits
    p_train = ROOT / "data/processed/sa_vedic_train.conllu"
    p_dev = ROOT / "data/processed/sa_vedic_dev.conllu"
    p_test = ROOT / "data/processed/sa_vedic_test.conllu"

    vedic_train = parse_conllu(p_train)
    vedic_dev = parse_conllu(p_dev)
    vedic_test = parse_conllu(p_test)

    if not vedic_dev:
        n_split = max(100, int(len(vedic_train) * 0.1))
        vedic_dev = vedic_train[-n_split:]
        vedic_train = vedic_train[:-n_split]
    if not vedic_test:
        n_split = max(100, int(len(vedic_dev) * 0.5))
        vedic_test = vedic_dev[-n_split:]
        vedic_dev = vedic_dev[:-n_split]

    log.info(f"UD Vedic splits loaded: Train={len(vedic_train):,}, Dev={len(vedic_dev):,}, Test={len(vedic_test):,}")

    # 2. Majority & Unigram-majority baselines on test split
    baseline_results = compute_majority_baselines(vedic_train, vedic_test)
    log.info(f"Majority Baseline: Acc={baseline_results['majority_class']['accuracy']:.4f}, Macro-F1={baseline_results['majority_class']['macro_f1']:.4f}")
    log.info(f"Unigram Majority Baseline: Acc={baseline_results['unigram_majority']['accuracy']:.4f}, Macro-F1={baseline_results['unigram_majority']['macro_f1']:.4f}")

    # 3. Stream Monolingual Text for MLM (fast line streamer)
    mlm_train_p = ROOT / "data/processed/mlm_train.txt"
    if not mlm_train_p.exists():
        mlm_train_p = ROOT / "data/demo/monolingual.txt"

    mlm_lines = []
    with mlm_train_p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                mlm_lines.append(line)
            if len(mlm_lines) >= args.max_mlm_lines:
                break

    mlm_valid_p = ROOT / "data/processed/mlm_valid.txt"
    mlm_val_lines = []
    if mlm_valid_p.exists():
        with mlm_valid_p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    mlm_val_lines.append(line)
                if len(mlm_val_lines) >= 500:
                    break
    if not mlm_val_lines:
        mlm_val_lines = mlm_lines[-100:]

    log.info(f"Monolingual lines for MLM: Train={len(mlm_lines):,}, Val={len(mlm_val_lines):,}")

    # 4. Vocabulary Construction
    vocab_path = ck_dir / "vocab.json"
    vocab_corpus = mlm_lines + [" ".join(f) for f, _ in vedic_train[:5000]]
    vocab = build_vocab(vocab_corpus, str(vocab_path), limit=12000)
    tok = SanskritTokenizer(str(vocab_path))
    vocab_size = tok.vocab_size
    log.info(f"Vocabulary ready: {vocab_size:,} tokens (saved to {vocab_path})")

    # 5. Dataloaders
    train_sents_bounded = vedic_train[:args.max_morph_sents]
    morph_train_loader = DataLoader(MorphDataset(train_sents_bounded, tok), batch_size=args.batch_size, shuffle=True)
    morph_dev_loader = DataLoader(MorphDataset(vedic_dev[:500], tok), batch_size=args.batch_size, shuffle=False)
    morph_test_loader = DataLoader(MorphDataset(vedic_test[:500], tok), batch_size=args.batch_size, shuffle=False)

    mlm_train_loader = DataLoader(MLMDataset(mlm_lines, tok), batch_size=args.batch_size, shuffle=True)
    mlm_val_loader = DataLoader(MLMDataset(mlm_val_lines, tok), batch_size=args.batch_size, shuffle=False)

    # -----------------------------------------------------------------------
    # 6. Baseline 2: No-Pretraining Transformer (Trained directly on UD Vedic)
    # -----------------------------------------------------------------------
    log.info("\n=== Training Baseline 2: No-Pretraining Transformer ===")
    model_no_pretrain = SanskritEncoder(vocab_size, len(TAGS)).to(device)
    no_pretrain_metrics = train_morphology_head(
        model_no_pretrain,
        morph_train_loader,
        morph_dev_loader,
        morph_test_loader,
        device,
        epochs=args.epochs_morph,
        lr=4e-4,
        name="No-Pretrain",
    )
    torch.save({
        "model": model_no_pretrain.state_dict(),
        "vocab_size": vocab_size,
        "tags": TAGS,
        "metrics": no_pretrain_metrics,
    }, ck_dir / "no_pretrain_baseline.pt")
    log.info(f"No-Pretraining Test Results: Acc={no_pretrain_metrics['accuracy']:.4f}, Macro-F1={no_pretrain_metrics['macro_f1']:.4f}")

    # -----------------------------------------------------------------------
    # 7. Stage 1: Masked Language Modeling Pretraining
    # -----------------------------------------------------------------------
    log.info("\n=== Stage 1: Masked Language Model (MLM) Pretraining ===")
    model = SanskritEncoder(vocab_size, len(TAGS)).to(device)
    opt_mlm = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    loss_fn_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    final_mlm_val_loss = 0.0
    final_mlm_val_ppl = 0.0

    for ep in range(args.epochs_mlm):
        model.train()
        total_mlm_loss, steps = 0.0, 0
        t0 = time.time()
        for ids in mlm_train_loader:
            ids = ids.to(device)
            labels = ids.clone()
            mask = (torch.rand(ids.shape, device=device) < 0.15) & (ids != 0)
            labels[~mask] = -100
            masked = ids.clone()
            masked[mask] = MASK_ID

            out = model(masked, (ids != 0).long(), mlm=True, morph=False, pool=False)
            loss = loss_fn_mlm(out["mlm_logits"].reshape(-1, vocab_size), labels.reshape(-1))

            opt_mlm.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_mlm.step()

            total_mlm_loss += loss.item()
            steps += 1

        val_loss, val_ppl = eval_mlm(model, mlm_val_loader, device, vocab_size)
        final_mlm_val_loss = val_loss
        final_mlm_val_ppl = val_ppl
        elapsed = time.time() - t0
        log.info(f"MLM Epoch {ep+1}/{args.epochs_mlm} ({elapsed:.1f}s) | Train Loss: {total_mlm_loss/max(1, steps):.4f} | Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")

    torch.save({
        "model": model.state_dict(),
        "vocab_size": vocab_size,
        "tags": TAGS,
        "val_loss": final_mlm_val_loss,
        "val_ppl": final_mlm_val_ppl,
    }, ck_dir / "pretrained_mlm.pt")

    # -----------------------------------------------------------------------
    # 8. Stage 2: Supervised Morphology Fine-Tuning
    # -----------------------------------------------------------------------
    log.info("\n=== Stage 2: Supervised Morphology Fine-Tuning ===")
    finetuned_metrics = train_morphology_head(
        model,
        morph_train_loader,
        morph_dev_loader,
        morph_test_loader,
        device,
        epochs=args.epochs_morph,
        lr=3e-4,
        name="Pretrained-Transformer",
    )
    log.info(f"Pretrained + Fine-Tuned Test Results: Acc={finetuned_metrics['accuracy']:.4f}, Macro-F1={finetuned_metrics['macro_f1']:.4f}")

    # 9. Save Checkpoints
    current_checkpoint = {
        "model": model.state_dict(),
        "vocab_size": vocab_size,
        "tags": TAGS,
        "config": {"hidden": model.hidden, "max_len": model.max_len, "num_tags": model.num_tags},
        "test_metrics": finetuned_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(current_checkpoint, ck_dir / "current.pt")
    torch.save(current_checkpoint, ck_dir / "best.pt")

    # Baseline F1 for self-learning guard
    baseline_f1_info = {
        "f1": finetuned_metrics["macro_f1"],
        "accuracy": finetuned_metrics["accuracy"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (ck_dir / "baseline_morph_f1.json").write_text(json.dumps(baseline_f1_info, indent=2), encoding="utf-8")

    # 10. Compile Complete Results Document
    acc_gain = round(finetuned_metrics["accuracy"] - no_pretrain_metrics["accuracy"], 4)
    f1_gain = round(finetuned_metrics["macro_f1"] - no_pretrain_metrics["macro_f1"], 4)

    results_data = {
        "metadata": {
            "project": "VYAKARANA-X",
            "device": device,
            "vocab_size": vocab_size,
            "model_parameters": model.num_parameters,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_splits": {
                "ud_vedic_train": len(vedic_train),
                "ud_vedic_dev": len(vedic_dev),
                "ud_vedic_test": len(vedic_test),
                "mlm_lines_used": len(mlm_lines),
            },
        },
        "baselines": {
            "majority_class": baseline_results["majority_class"],
            "unigram_majority": baseline_results["unigram_majority"],
            "no_pretraining_transformer": no_pretrain_metrics,
        },
        "pretrained_and_finetuned": {
            "morphology_test_accuracy": finetuned_metrics["accuracy"],
            "morphology_test_macro_f1": finetuned_metrics["macro_f1"],
            "morphology_test_weighted_f1": finetuned_metrics["weighted_f1"],
            "mlm_val_loss": final_mlm_val_loss,
            "mlm_val_perplexity": final_mlm_val_ppl,
        },
        "pretraining_impact": {
            "accuracy_delta_vs_no_pretrain": acc_gain,
            "macro_f1_delta_vs_no_pretrain": f1_gain,
            "accuracy_delta_vs_majority": round(finetuned_metrics["accuracy"] - baseline_results["majority_class"]["accuracy"], 4),
            "macro_f1_delta_vs_majority": round(finetuned_metrics["macro_f1"] - baseline_results["majority_class"]["macro_f1"], 4),
        },
    }

    results_file = metrics_dir / "results.json"
    results_file.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    log.info(f"\nFinal results compiled to {results_file}")

    # Training metrics JSON for backend / dashboard
    training_metrics_data = {
        "trained": True,
        "epochs_mlm": args.epochs_mlm,
        "epochs_morph": args.epochs_morph,
        "device": device,
        "vocab_size": vocab_size,
        "model_params": model.num_parameters,
        "mlm_val_loss": final_mlm_val_loss,
        "mlm_val_perplexity": final_mlm_val_ppl,
        "morphology_loss": finetuned_metrics["loss"],
        "morphology_accuracy": finetuned_metrics["accuracy"],
        "morphology_f1": finetuned_metrics["macro_f1"],
        "results": results_data,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (ck_dir / "training_metrics.json").write_text(json.dumps(training_metrics_data, indent=2), encoding="utf-8")
    log.info(f"Dashboard training metrics saved to {ck_dir / 'training_metrics.json'}")

    print("\n" + "=" * 60)
    print("VYAKARANA-X EVALUATION SUMMARY (HELD-OUT UD VEDIC TEST SPLIT)")
    print("=" * 60)
    print(f"1. Majority Class Baseline:        Acc = {baseline_results['majority_class']['accuracy']*100:.2f}%, Macro-F1 = {baseline_results['majority_class']['macro_f1']:.4f}")
    print(f"2. Unigram Majority Baseline:     Acc = {baseline_results['unigram_majority']['accuracy']*100:.2f}%, Macro-F1 = {baseline_results['unigram_majority']['macro_f1']:.4f}")
    print(f"3. No-Pretraining Transformer:     Acc = {no_pretrain_metrics['accuracy']*100:.2f}%, Macro-F1 = {no_pretrain_metrics['macro_f1']:.4f}")
    print(f"4. Pretrained + Fine-Tuned (Full): Acc = {finetuned_metrics['accuracy']*100:.2f}%, Macro-F1 = {finetuned_metrics['macro_f1']:.4f}")
    print(f"--> Delta over No-Pretraining:     +{(f1_gain)*100:.2f}% Macro-F1 (+{(acc_gain)*100:.2f}% Acc)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
