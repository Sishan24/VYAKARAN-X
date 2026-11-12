"""
VYAKARANA-X data preparation script.
Cleans and deduplicates monolingual text; organizes Universal Dependencies Vedic splits
(train, dev, test) and preserves Saamayik parallel splits.
"""
from pathlib import Path
import re, json, unicodedata, random, shutil

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
    train_txt = OUT / "mlm_train.txt"
    valid_txt = OUT / "mlm_valid.txt"

    train_count = 0
    valid_count = 0

    if train_txt.exists() and train_txt.stat().st_size > 1000000:
        print(f"Using existing prepared mlm_train.txt ({train_txt.stat().st_size:,} bytes)")
        train_count = sum(1 for line in train_txt.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
        if valid_txt.exists():
            valid_count = sum(1 for line in valid_txt.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
    else:
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
            train_txt.write_text("\n".join(train_rows), encoding="utf-8")
            valid_txt.write_text("\n".join(valid_rows), encoding="utf-8")
            train_count = len(train_rows)
            valid_count = len(valid_rows)
            print(f"Monolingual: {train_count:,} train / {valid_count:,} val lines")

    # Copy official UD Vedic splits (train, dev, test)
    ud_stats = {}
    for split in ["train", "dev", "test"]:
        raw_p = RAW / f"sa_vedic_{split}.conllu"
        out_p = OUT / f"sa_vedic_{split}.conllu"
        if raw_p.exists():
            if not out_p.exists() or out_p.stat().st_size != raw_p.stat().st_size:
                shutil.copy2(raw_p, out_p)
            stats = count_conllu_stats(out_p)
            ud_stats[split] = stats
            print(f"UD Vedic {split}: {stats['sentences']:,} sentences, {stats['tokens']:,} tokens -> {out_p.name}")

    # Preserve Saamayik splits
    saamayik_stats = {}
    for p in RAW.glob("saamayik_*.jsonl"):
        out_target = OUT / p.name
        if not out_target.exists() or out_target.stat().st_size != p.stat().st_size:
            shutil.copy2(p, out_target)
        count = sum(1 for line in out_target.read_text(encoding="utf-8").splitlines() if line.strip())
        saamayik_stats[p.stem] = count
        print(f"Preserved {p.name} ({count:,} pairs)")

    report = {
        "unique_monolingual_lines": train_count + valid_count,
        "mlm_train_lines": train_count,
        "mlm_valid_lines": valid_count,
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
