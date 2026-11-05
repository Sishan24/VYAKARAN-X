from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json, subprocess, sys
from grammar.paninian_engine import inspect_tokens, consistency_score
ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title="VYAKARANA-X API",version="1.0")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
class Sentence(BaseModel): text:str
@app.get("/api/health")
def health():
    return {"status":"online","project":"VYAKARANA-X","checkpoint":(ROOT/"checkpoints/current.pt").exists()}
@app.get("/api/stats")
def stats():
    def read(p):
        try:return json.loads(p.read_text(encoding="utf-8"))
        except:return {}
    return {"training":read(ROOT/"checkpoints/training_metrics.json"),"data":read(ROOT/"data/processed/data_report.json"),
            "self_learning":read(ROOT/"checkpoints/self_learning_report.json"),
            "checkpoint":(ROOT/"checkpoints/current.pt").exists()}
@app.post("/api/analyze")
def analyze(req:Sentence):
    tokens=[x for x in req.text.replace("।"," । ").split() if x]
    score,details=consistency_score(tokens)
    return {"input":req.text,"tokens":details,"grammar_compatibility":round(score,3),
            "notice":"Explainable suffix heuristics only; unresolved tokens remain ambiguous. This is not a complete Paninian parser."}
@app.post("/api/self-learn")
def self_learn():
    p=subprocess.run([sys.executable,"-m","self_learning.cycle","--max-pseudo","2000"],cwd=ROOT,capture_output=True,text=True,timeout=300)
    return {"ok":p.returncode==0,"output":p.stdout,"error":p.stderr}
