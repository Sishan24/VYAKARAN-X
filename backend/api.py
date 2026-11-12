"""
VYAKARANA-X FastAPI backend.
Serves:
- /api/health          : Liveness, checkpoint and vocab status
- /api/stats           : Training metrics, baselines, data report, self-learning history
- /api/results         : Full metrics/results.json baselines document
- /api/analyze         : Paninian rule-based grammar inspection & consistency
- /api/inference       : Neural model POS tagging & token representation
- /api/train           : Trigger training run in background
- /api/self-learn      : Trigger guarded self-learning cycle
- /api/pseudo-labels   : List generated pseudo-label candidates
- /api/review-queue    : List active learning human review candidates
"""
import json, logging, subprocess, sys
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grammar.paninian_engine import inspect_tokens, consistency_score, sandhi_check

log = logging.getLogger("vyakarana_api")

app = FastAPI(title="VYAKARANA-X API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy Model Cache
# ---------------------------------------------------------------------------
_model_cache: dict = {}

def _load_model():
    """Load model once and cache."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["tok"], _model_cache["tags"]
    ck = ROOT / "checkpoints"
    cp = ck / "current.pt"
    vp = ck / "vocab.json"
    if not cp.exists() or not vp.exists():
        return None, None, None
    try:
        from models.network import SanskritEncoder
        from data_utils.vocab import SanskritTokenizer
        state = torch.load(cp, map_location="cpu", weights_only=False)
        tags = state.get("tags", ["PAD", "UNK", "NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "NUM", "PART", "PUNCT", "X"])
        model = SanskritEncoder(state["vocab_size"], len(tags))
        model.load_state_dict(state["model"])
        model.eval()
        tok = SanskritTokenizer(str(vp))
        _model_cache.update({"model": model, "tok": tok, "tags": tags})
        return model, tok, tags
    except Exception as e:
        log.error("Model load error: %s", e)
        return None, None, None


def _read_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Error reading JSON from {path}: {e}")
    return {}


def _read_jsonl(path: Path, limit: int = 100) -> list:
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= limit:
                break
    except Exception as e:
        log.warning(f"Error reading JSONL from {path}: {e}")
    return out


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------
class Sentence(BaseModel):
    text: str

class TrainRequest(BaseModel):
    epochs_mlm: int = 1
    epochs_morph: int = 1
    smoke: bool = True

class SelfLearnRequest(BaseModel):
    max_pseudo: int = 500
    simulate_rollback: bool = False


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    ck = ROOT / "checkpoints"
    return {
        "status": "online",
        "project": "VYAKARANA-X",
        "checkpoint": (ck / "current.pt").exists(),
        "best_checkpoint": (ck / "best.pt").exists(),
        "vocab_ready": (ck / "vocab.json").exists(),
        "results_ready": (ROOT / "metrics" / "results.json").exists(),
    }


@app.get("/api/results")
def results():
    return _read_json(ROOT / "metrics" / "results.json")


@app.get("/api/stats")
def stats():
    ck = ROOT / "checkpoints"
    pseudo_lines = _read_jsonl(ck / "pseudo_labels.jsonl", limit=10000)
    history = _read_json(ck / "self_learning_history.json")
    if not isinstance(history, list):
        history = []
    return {
        "training": _read_json(ck / "training_metrics.json"),
        "results": _read_json(ROOT / "metrics" / "results.json"),
        "data": _read_json(ROOT / "data/processed/data_report.json"),
        "self_learning": _read_json(ck / "self_learning_report.json"),
        "self_learning_history": history,
        "checkpoint": (ck / "current.pt").exists(),
        "pseudo_count": len(pseudo_lines),
    }


@app.post("/api/analyze")
def analyze(req: Sentence):
    try:
        text = req.text.replace("\u0964", " \u0964 ").replace("\u0965", " \u0965 ")
        tokens = [t for t in text.split() if t]
        score, details = consistency_score(tokens)
        has_sandhi = sandhi_check(req.text)
        return {
            "input": req.text,
            "tokens": details,
            "grammar_compatibility": round(score, 3),
            "sandhi_detected": has_sandhi,
            "notice": "Explainable suffix heuristics only. Unresolved tokens retain morphological ambiguity.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inference")
def inference(req: Sentence):
    model, tok, tags = _load_model()
    if model is None:
        return {"error": "No trained checkpoint found. Train first: python train.py --smoke"}
    try:
        ids_list = tok.encode(req.text, max_length=128)
        if not ids_list:
            return {"error": "Could not tokenize input."}
        ids_t = torch.tensor([ids_list])
        mask_t = torch.ones_like(ids_t)

        with torch.no_grad():
            out = model(ids_t, mask_t, mlm=False, morph=True, pool=True)
            probs = out["morph_logits"].softmax(-1)[0]

        result_tokens = []
        display_toks = tok.tokenize(req.text)
        for i, (tok_str, prob_vec) in enumerate(zip(display_toks, probs)):
            top3 = prob_vec.topk(min(3, len(tags)))
            result_tokens.append({
                "token": tok_str,
                "top_tags": [
                    {"tag": tags[idx.item()], "prob": round(float(val.item()), 4)}
                    for val, idx in zip(top3.values, top3.indices)
                ],
            })

        sent_norm = float(out["sentence_vec"].norm().item()) if out.get("sentence_vec") is not None else 0.0

        return {
            "input": req.text,
            "tokens": result_tokens,
            "sentence_vec_norm": round(sent_norm, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
def train(req: TrainRequest):
    try:
        cmd = [
            sys.executable, "-u", str(ROOT / "train.py"),
            "--epochs-mlm", str(req.epochs_mlm),
            "--epochs-morph", str(req.epochs_morph),
        ]
        if req.smoke:
            cmd.append("--smoke")
        proc = subprocess.Popen(cmd, cwd=str(ROOT))
        return {"started": True, "pid": proc.pid, "command": " ".join(cmd)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/self-learn")
def self_learn(req: SelfLearnRequest):
    try:
        cmd = [
            sys.executable, "-u", "-m", "self_learning.cycle",
            "--max-pseudo", str(req.max_pseudo),
        ]
        if req.simulate_rollback:
            cmd.append("--simulate-rollback")
        proc = subprocess.Popen(cmd, cwd=str(ROOT))
        return {"started": True, "pid": proc.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pseudo-labels")
def pseudo_labels():
    return _read_jsonl(ROOT / "checkpoints/pseudo_labels.jsonl", limit=100)


@app.get("/api/review-queue")
def review_queue():
    return _read_jsonl(ROOT / "checkpoints/active_learning_queue.jsonl", limit=50)


@app.get("/api/morphology-eval")
def morphology_eval():
    m = _read_json(ROOT / "checkpoints/training_metrics.json")
    res = _read_json(ROOT / "metrics/results.json")
    return {
        "morphology_accuracy": m.get("morphology_accuracy"),
        "morphology_f1": m.get("morphology_f1"),
        "mlm_val_loss": m.get("mlm_val_loss"),
        "mlm_val_perplexity": m.get("mlm_val_perplexity"),
        "trained": m.get("trained", False),
        "results": res,
    }
