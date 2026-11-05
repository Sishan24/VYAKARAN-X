import argparse, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models.network import SanskritEncoder
from data_utils.vocab import build_vocab, SanskritTokenizer
ROOT=Path(__file__).resolve().parent
MASK=2
TAGS=["PAD","UNK","NOUN","PROPN","VERB","ADJ","ADV","PRON","DET","ADP","CCONJ","SCONJ","NUM","PART","PUNCT","X"]
tag2id={x:i for i,x in enumerate(TAGS)}

def parse_conllu(path):
    sentences=[]; forms=[]; tags=[]
    if not path.exists(): return sentences
    for line in path.read_text(encoding="utf-8").splitlines()+[""]:
        if not line:
            if forms: sentences.append((forms,tags)); forms=[]; tags=[]
            continue
        if line.startswith("#"): continue
        c=line.split("\t")
        if len(c)==10 and "-" not in c[0] and "." not in c[0]:
            forms.append(c[1]); tags.append(c[3] if c[3] in tag2id else "X")
    return sentences

class TextDS(Dataset):
    def __init__(self, lines,tok,maxlen=128):
        self.x=[tok.encode(x,maxlen) for x in lines if tok.encode(x,maxlen)]
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        ids=self.x[i][:128]; return torch.tensor(ids+[0]*(128-len(ids)))
class MorphDS(Dataset):
    def __init__(self,sentences,tok,maxlen=128):
        self.rows=[]
        for forms,tags in sentences:
            ids=tok.encode(" ".join(forms),maxlen)
            labels=[tag2id.get(t,1) for t in tags][:maxlen]
            n=min(len(ids),len(labels)); ids=ids[:n]; labels=labels[:n]
            if n: self.rows.append((ids,labels))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        ids,labs=self.rows[i]; n=len(ids)
        return torch.tensor(ids+[0]*(128-n)),torch.tensor(labs+[-100]*(128-n))
def load_lines(smoke=False, cap=100000):
    p=ROOT/"data/processed/mlm_train.txt"
    if not p.exists(): p=ROOT/"data/demo/monolingual.txt"
    lines=p.read_text(encoding="utf-8").splitlines()
    return lines[:min(cap,2000 if smoke else cap)]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--max-lines",type=int,default=100000); ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args(); random.seed(42); torch.manual_seed(42)
    lines=load_lines(args.smoke,args.max_lines)
    morph_sentences=parse_conllu(ROOT/"data/processed/sa_vedic_train.conllu")
    # Include UD forms in the vocabulary so annotated tokens are not all OOV.
    vocab_path=ROOT/"checkpoints/vocab.json"
    vocab=build_vocab(lines+[" ".join(x[0]) for x in morph_sentences],vocab_path,limit=12000)
    tok=SanskritTokenizer(vocab_path)
    mlm=DataLoader(TextDS(lines,tok),batch_size=8 if args.smoke else 16,shuffle=True)
    morph=DataLoader(MorphDS(morph_sentences,tok),batch_size=8,shuffle=True) if morph_sentences else []
    model=SanskritEncoder(len(vocab),len(TAGS))
    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    loss_fn=nn.CrossEntropyLoss(ignore_index=-100)
    model.train(); final_mlm=0; final_morph=0
    for ep in range(args.epochs):
        total=0; steps=0
        for ids in mlm:
            ids=ids.to(device); labels=ids.clone()
            mask=(torch.rand(ids.shape,device=device)<.15)&(ids!=0)
            labels[~mask]=-100
            masked=ids.clone(); masked[mask]=MASK
            out=model(masked,(ids!=0).long())
            loss=loss_fn(out["mlm_logits"].reshape(-1,len(vocab)),labels.reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=loss.item(); steps+=1
        final_mlm=total/max(1,steps)
        mtotal=0; msteps=0
        if morph:
            for ids,labs in morph:
                ids=ids.to(device); labs=labs.to(device)
                out=model(ids,(ids!=0).long())
                loss=loss_fn(out["morph_logits"].reshape(-1,len(TAGS)),labs.reshape(-1))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                mtotal+=loss.item(); msteps+=1
        final_morph=mtotal/max(1,msteps) if msteps else None
        print(f"epoch {ep+1}/{args.epochs} | MLM loss={final_mlm:.4f} | morphology loss={final_morph if final_morph is not None else 'UD unavailable'}")
    ck=ROOT/"checkpoints"; ck.mkdir(exist_ok=True)
    torch.save({"model":model.state_dict(),"vocab_size":len(vocab),"tags":TAGS,"config":{"hidden":256,"layers":4,"heads":4}},ck/"current.pt")
    (ck/"training_metrics.json").write_text(json.dumps({"trained":True,"lines":len(lines),"ud_morph_sentences":len(morph_sentences),"epochs":args.epochs,"device":device,"mlm_loss":final_mlm,"morphology_loss":final_morph},indent=2))
    print("Saved checkpoints/current.pt")
if __name__=="__main__": main()
