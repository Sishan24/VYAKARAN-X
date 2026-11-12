"""
VYAKARANA-X data downloader.
Downloads:
- Saamayik parallel corpus (HuggingFace mirror)
- Universal Dependencies Vedic Sanskrit official treebank splits (train, dev, test)
- Sansar Sanskrit monolingual corpus sample (MuseMesh/sansar-sanskrit-corpus)
"""
import argparse, json, sys, time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

UD_BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Sanskrit-Vedic/master"
UD_SPLITS = ["train", "dev", "test"]

def _fetch_url(url: str, timeout: int = 90, retries: int = 2) -> bytes:
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "VYAKARANA-X/2.0 research downloader"})
            with urlopen(req, timeout=timeout) as r:
                return r.read()
        except URLError as e:
            if attempt < retries:
                print(f"  Retry {attempt+1}/{retries} after: {e}")
                time.sleep(2 ** attempt)
            else:
                raise

def download_ud_vedic():
    print("Downloading Universal Dependencies Vedic Sanskrit official splits...")
    summary = {}
    for split in UD_SPLITS:
        target = RAW / f"sa_vedic_{split}.conllu"
        if target.exists() and target.stat().st_size > 100000:
            print(f"  Found existing {target.name} ({target.stat().st_size:,} bytes)")
            summary[f"ud_vedic_{split}_bytes"] = target.stat().st_size
            continue
        url = f"{UD_BASE_URL}/sa_vedic-ud-{split}.conllu"
        try:
            data = _fetch_url(url)
            target.write_bytes(data)
            print(f"  Downloaded {target.name} ({len(data):,} bytes)")
            summary[f"ud_vedic_{split}_bytes"] = len(data)
        except Exception as e:
            print(f"  WARNING: Failed to download {split} split from {url}: {e}")
            summary[f"ud_vedic_{split}_error"] = str(e)
    return summary

def download_saamayik():
    print("Downloading Saamayik parallel corpus (HuggingFace)...")
    summary = {}
    try:
        from datasets import load_dataset
        ds = load_dataset("acomquest/Saamayik")
        for split in ds:
            out_path = RAW / f"saamayik_{split}.jsonl"
            ds[split].to_json(str(out_path), force_ascii=False)
            summary[f"saamayik_{split}"] = len(ds[split])
            print(f"  Split '{split}': {len(ds[split]):,} rows -> {out_path.name}")
        return summary
    except Exception as e:
        print("  WARNING: Saamayik download warning:", e)
        # Check if already present
        for split in ["train", "validation", "test", "test_ood"]:
            p = RAW / f"saamayik_{split}.jsonl"
            if p.exists():
                count = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
                summary[f"saamayik_{split}"] = count
        return summary

def download_sansar(max_lines: int):
    target = RAW / "sansar_sample.txt"
    if target.exists() and target.stat().st_size > 1000000:
        lines_count = sum(1 for line in target.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
        print(f"  Found existing sansar_sample.txt with {lines_count:,} lines ({target.stat().st_size:,} bytes)")
        return {"sansar_lines": lines_count}
    print(f"Streaming Sansar Sanskrit corpus (cap={max_lines:,})...")
    try:
        from datasets import load_dataset
        ds = load_dataset("MuseMesh/sansar-sanskrit-corpus", split="train", streaming=True)
        rows = []
        for row in ds:
            val = next((row[k] for k in ("text", "content", "sentence", "sanskrit") if isinstance(row.get(k), str)), None)
            if val:
                rows.append(val)
            if len(rows) >= max_lines:
                break
        with target.open("w", encoding="utf-8") as f:
            for r in rows:
                clean_r = " ".join(str(r).split())
                if clean_r:
                    f.write(clean_r + "\n")
        print(f"  Saved {len(rows):,} Sansar lines -> {target.name}")
        return {"sansar_lines": len(rows)}
    except Exception as e:
        print("  WARNING: Sansar stream warning:", e)
        return {"sansar_error": str(e)}

def main():
    p = argparse.ArgumentParser(description="Download VYAKARANA-X training data")
    p.add_argument("--max-lines", type=int, default=100000)
    p.add_argument("--skip-sansar", action="store_true")
    p.add_argument("--skip-saamayik", action="store_true")
    p.add_argument("--skip-ud", action="store_true")
    args = p.parse_args()

    summary = {}
    if not args.skip_ud:
        summary.update(download_ud_vedic())
    if not args.skip_saamayik:
        summary.update(download_saamayik())
    if not args.skip_sansar:
        summary.update(download_sansar(args.max_lines))

    (RAW / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDownload summary saved to data/raw/download_summary.json:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
