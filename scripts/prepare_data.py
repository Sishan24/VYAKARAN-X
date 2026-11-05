from pathlib import Path
import re, json, unicodedata, random
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"; OUT=ROOT/"data"/"processed"
OUT.mkdir(parents=True,exist_ok=True)
DEV=re.compile(r"[\u0900-\u097F]")
def clean(s):
    s=unicodedata.normalize("NFC",str(s)).replace("\ufeff","")
    s=" ".join(s.split())
    if len(s)<3 or not DEV.search(s): return None
    # discard lines with a very low Devanagari proportion (likely metadata/noise)
    letters=[c for c in s if c.isalpha()]
    if letters and sum("\u0900"<=c<="\u097F" for c in letters)/len(letters)<.35: return None
    return s
def main():
    candidates=[]
    for p in [RAW/"sansar_sample.txt", ROOT/"data/demo/monolingual.txt"]:
        if p.exists(): candidates += p.read_text(encoding="utf-8").splitlines()
    seen=set(); rows=[]
    for x in candidates:
        x=clean(x)
        if x and x not in seen: seen.add(x); rows.append(x)
    random.Random(42).shuffle(rows)
    if rows:
        n=max(1,int(len(rows)*.95))
        (OUT/"mlm_train.txt").write_text("\n".join(rows[:n]),encoding="utf-8")
        (OUT/"mlm_valid.txt").write_text("\n".join(rows[n:] or rows[-1:]),encoding="utf-8")
    # preserve the official UD CoNLL-U for morphology parser
    if (RAW/"sa_vedic_train.conllu").exists():
        (OUT/"sa_vedic_train.conllu").write_bytes((RAW/"sa_vedic_train.conllu").read_bytes())
    # Saamayik train/validation/test are preserved as separate files, never merged.
    for p in RAW.glob("saamayik_*.jsonl"):
        (OUT/p.name).write_bytes(p.read_bytes())
    (OUT/"data_report.json").write_text(json.dumps({"unique_monolingual_lines":len(rows),"sources":["Sansar sample if downloaded","UD Vedic if downloaded","demo fallback only for smoke test"]},indent=2),encoding="utf-8")
    print(f"Prepared {len(rows)} unique Sanskrit lines at {OUT}")
if __name__=="__main__": main()
