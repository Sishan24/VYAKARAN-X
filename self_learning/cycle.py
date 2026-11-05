"""Confidence-filtered pseudo-label + guarded continual update demonstration."""
import json, random, shutil
from pathlib import Path
import torch
from torch import nn
from models.network import SanskritEncoder
from datasets.vocab import SanskritTokenizer
from grammar.paninian_engine import consistency_score
ROOT=Path(__file__).resolve().parents[1]
def run(max_pseudo=2000):
    ck=ROOT/"checkpoints"; current=ck/"current.pt"; vocabfile=ck/"vocab.json"
    if not current.exists() or not vocabfile.exists():
        raise SystemExit("Train first: python train.py --epochs 1")
    device="cuda" if torch.cuda.is_available() else "cpu"
    state=torch.load(current,map_location=device,weights_only=False)
    tok=SanskritTokenizer(vocabfile); tags=state.get("tags",["PAD","UNK","NOUN","PROPN","VERB","ADJ","ADV","PRON","DET","ADP","CCONJ","SCONJ","NUM","PART","PUNCT","X"])
    model=SanskritEncoder(state["vocab_size"],len(tags)).to(device); model.load_state_dict(state["model"]); model.eval()
    source=ROOT/"data/processed/mlm_train.txt"
    lines=source.read_text(encoding="utf-8").splitlines() if source.exists() else []
    pseudo=[]; review=[]
    for line in lines[:max_pseudo]:
        toks=tok.tokenize(line); ids=tok.encode(line)
        if not ids: continue
        with torch.no_grad():
            out=model(torch.tensor([ids],device=device),torch.ones((1,len(ids)),device=device,dtype=torch.long))
            probs=out["morph_logits"].softmax(-1).max(-1).values[0]
            confidence=float(probs.mean().item())
        grammar,_=consistency_score(toks)
        # High threshold + grammar plausibility; model confidence is not treated as truth.
        if confidence>=.72 and grammar>=.65: pseudo.append({"text":line,"confidence":confidence,"grammar_score":grammar})
        elif confidence<.72: review.append({"text":line,"confidence":confidence,"reason":"uncertain prediction"})
    (ck/"pseudo_labels.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in pseudo),encoding="utf-8")
    (ck/"active_learning_queue.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in review[:500]),encoding="utf-8")
    # Conservative candidate update: retain weights unless a real supervised update is possible.
    # This prevents pretending pseudo-label creation alone improved the model.
    report={"pseudo_candidates":len(pseudo),"human_review_queue":min(500,len(review)),
            "promoted":False,"reason":"No independent validated evaluation improvement has been demonstrated. Candidate promotion is intentionally blocked until a supervised validation run is available.",
            "status":"pseudo-label buffer generated; checkpoint unchanged"}
    (ck/"self_learning_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--max-pseudo",type=int,default=2000); a=p.parse_args(); run(a.max_pseudo)
