"""
VYAKARANA-X data preparation script.
Cleans and deduplicates monolingual text; organizes Universal Dependencies Vedic splits
(train, dev, test) and preserves Saamayik parallel splits.
"""
from pathlib import Path
import re, json, unicodedata, random

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

DEV = re.compile(r"[\u0900-\u097F]")

def clean(s):
    s = unicodedata.normalize("NFC", str(s)).replace("\ufeff", "")
    s = " ".join(s.split())
    if len(s) < 3 or not DEV.search(s):
        return None
    # Discard lines with a very low Devanagari proportion (metadata/foreign noise)
    letters = [c for c in s if c.isalpha()]
    if letters and sum("\u0900" <= c <= "\u097F" for c in letters) / len(letters) < 0.35:
        return None
    return s

def count_conllu_stats(p: Path):
    if not p.exists():
        return {"sentences": 0, "tokens": 0}
    sentences = 0
    tokens = 0
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                sentences += 1
            continue
        c = line.split("\t")
        if len(c) == 10 and "-" not in c[0] and "." not in c[0]:
            tokens += 1
    return {"sentences": sentences, "tokens": tokens}

def main():
    print("Preparing VYAKARANA-X datasets...")
    candidates = []
    for p in [RAW / "sansar_sample.txt", ROOT / "data/demo/monolingual.txt"]:
        if p.exists():
            candidates += p.read_text(encoding="utf-8", errors="ignore").splitlines()

    seen = set()
    rows = []
    for x in candidates:
        x = clean(x)
        if x and x not in seen:
            seen.add(x)
            rows.append(x)

    random.Random(42).shuffle(rows)
    if rows:
        n = max(1, int(len(rows) * 0.95))
        train_rows = rows[:n]
        valid_rows = rows[n:] if len(rows) > n else rows[-1:]
        (OUT / "mlm_train.txt").write_text("\n".join(train_rows), encoding="utf-8")
        (OUT / "mlm_valid.txt").write_text("\n".join(valid_rows), encoding="utf-8")
        print(f"Monolingual: {len(train_rows):,} train / {len(valid_rows):,} val lines")

    # Copy official UD Vedic splits (train, dev, test)
    ud_stats = {}
    for split in ["train", "dev", "test"]:
        raw_p = RAW / f"sa_vedic_{split}.conllu"
        out_p = OUT / f"sa_vedic_{split}.conllu"
        if raw_p.exists():
            out_p.write_bytes(raw_p.read_bytes())
            stats = count_conllu_stats(out_p)
            ud_stats[split] = stats
            print(f"UD Vedic {split}: {stats['sentences']:,} sentences, {stats['tokens']:,} tokens -> {out_p.name}")

    # Preserve Saamayik splits
    saamayik_stats = {}
    for p in RAW.glob("saamayik_*.jsonl"):
        out_target = OUT / p.name
        out_target.write_bytes(p.read_bytes())
        count = sum(1 for line in out_target.read_text(encoding="utf-8").splitlines() if line.strip())
        saamayik_stats[p.stem] = count
        print(f"Preserved {p.name} ({count:,} pairs)")

    report = {
        "unique_monolingual_lines": len(rows),
        "mlm_train_lines": len(rows[:n]) if rows else 0,
        "mlm_valid_lines": len(rows[n:]) if rows else 0,
        "ud_vedic_splits": ud_stats,
        "saamayik_splits": saamayik_stats,
        "sources": [
            "Sansar Sanskrit Corpus (MuseMesh/sansar-sanskrit-corpus)",
            "Universal Dependencies Vedic Sanskrit Treebank (official train, dev, test splits)",
            "Saamayik Sanskrit-English Parallel Corpus (acomquest/Saamayik)"
        ]
    }
    (OUT / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Data preparation complete. Report saved to {OUT / 'data_report.json'}")

if __name__ == "__main__":
    main()
