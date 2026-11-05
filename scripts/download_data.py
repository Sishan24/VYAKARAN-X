from pathlib import Path
import argparse, json, sys
from datasets import load_dataset

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"
RAW.mkdir(parents=True,exist_ok=True)

def save_text(path, rows):
    with path.open("w",encoding="utf-8") as f:
        for x in rows:
            x=" ".join(str(x).split())
            if x: f.write(x+"\n")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--max-lines",type=int,default=100000)
    p.add_argument("--skip-sansar",action="store_true")
    args=p.parse_args()
    summary={}
    # Saamayik: use published train split only; preserve held-out splits separately.
    try:
        ds=load_dataset("acomquest/Saamayik")
        for split in ds:
            ds[split].to_json(str(RAW/f"saamayik_{split}.jsonl"),force_ascii=False)
            summary[f"saamayik_{split}"]=len(ds[split])
        print("Downloaded Saamayik splits.")
    except Exception as e:
        print("Saamayik download warning:",e)
    # UD Vedic via official GitHub raw CoNLL-U.
    try:
        from urllib.request import urlopen
        url="https://raw.githubusercontent.com/UniversalDependencies/UD_Sanskrit-Vedic/master/sa_vedic-ud-train.conllu"
        data=urlopen(url,timeout=60).read()
        (RAW/"sa_vedic_train.conllu").write_bytes(data)
        summary["ud_vedic_bytes"]=len(data)
    except Exception as e:
        print("UD Vedic download warning:",e)
    # Sansar is streamed and capped; dataset config names are maintained upstream.
    if not args.skip_sansar:
        try:
            ds=load_dataset("MuseMesh/sansar-sanskrit-corpus",split="train",streaming=True)
            rows=[]
            for row in ds:
                # The corpus card may evolve; select likely text field robustly.
                val=next((row[k] for k in ("text","content","sentence","sanskrit") if isinstance(row.get(k),str)),None)
                if val: rows.append(val)
                if len(rows)>=args.max_lines: break
            save_text(RAW/"sansar_sample.txt",rows)
            summary["sansar_lines"]=len(rows)
            print(f"Saved {len(rows):,} Sansar records.")
        except Exception as e:
            print("Sansar stream warning:",e)
            print("If the upstream dataset now requires a config, consult its dataset card and update this loader.")
    (RAW/"download_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Download summary:",summary)
if __name__=="__main__": main()
