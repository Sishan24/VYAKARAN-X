# VYAKARANA-X — Autonomous Sanskrit Intelligence

A runnable research prototype for Sanskrit language learning: a compact Transformer, masked-language pretraining, morphology supervision, grammar-aware analysis, confidence-filtered pseudo-labelling, and guarded continual updates.

> **Research honesty:** this is a practical 30M-scale-*ambition* prototype, not the 150M architecture described in the concept document. The included model is deliberately small enough to experiment with on a CPU or modest GPU. Translation quality is not claimed. The first executable milestone is Sanskrit representation + morphology + self-training. The parallel Saamayik corpus is downloaded and prepared as a future translation extension.

## What is implemented

- PyTorch Transformer encoder trained from scratch with masked-language modelling (MLM).
- Sanskrit-aware character/subword vocabulary built from downloaded Devanagari data.
- Token-level morphology head trained from Universal Dependencies Vedic Sanskrit CoNLL-U.
- Paninian-inspired *transparent heuristic* checks for a few high-confidence nominal endings (not a complete Aṣṭādhyāyī engine).
- Confidence-filtered pseudo-labeling on unlabelled text.
- Continual-learning cycle: replay buffer, baseline evaluation, candidate checkpoint, promotion only if held-out score does not regress; otherwise rollback.
- FastAPI backend and Tailwind + React dashboard with live corpus/training metrics.
- Dataset manifest, attribution, and license notes.

## Data sources

The downloader uses only these subsets by default:

1. **Sansar Sanskrit Corpus** — large Devanagari monolingual corpus; the project page describes 252M words, source-level attribution, deduplication, and redistribution filtering. The script streams a bounded sample (default 100,000 lines), not the entire corpus. Check the upstream per-record license and attribution before redistributing a downloaded copy. [Dataset card](https://huggingface.co/datasets/MuseMesh/sansar-sanskrit-corpus)
2. **UD Sanskrit Vedic** — manually validated morphosyntactic annotations (about 4,000 sentences / 27,000 words). License: CC BY-SA 4.0. [Treebank](https://universaldependencies.org/treebanks/sa_vedic/index.html)
3. **Saamayik** — around 53,000 contemporary prose Sanskrit-English parallel pairs; the Hugging Face mirror declares MIT. Used for corpus inspection / future translation experiments, not silently mixed into the MLM objective. [Dataset card](https://huggingface.co/datasets/acomquest/Saamayik) · [Paper](https://aclanthology.org/2024.lrec-main.1245/)

The larger `Anamavajra-Labs/sanskrit-corpus` collection is **not** automatically consumed because its card explicitly mixes permissive, ShareAlike, non-commercial, research-only, and unclear-license subsets. If you use it later, filter every row by its license field and preserve source attribution.

## Quick start (Windows / PowerShell)

1. Install **Python 3.12** and **Node.js 20+**.
2. Extract this folder.
3. Double-click `START_WINDOWS.bat`.
4. The first launch creates `.venv`, installs Python dependencies, downloads the approved starter datasets, prepares them, and starts the API + frontend.
5. Open the local address shown in the terminal (normally `http://localhost:5173`).

The first setup needs internet access and may take several minutes. If the dataset download is unavailable, the dashboard still starts in demo mode; the bundled `data/demo/` samples are for smoke-testing only and are **not** sufficient for research claims.

### Manual setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
```

In terminal 1:
```powershell
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```
In terminal 2:
```powershell
cd frontend
npm install
npm run dev
```

## Train the actual model

```powershell
python train.py --epochs 3 --max-lines 100000
```

For a quick smoke test:
```powershell
python train.py --epochs 1 --max-lines 2000 --smoke
```

Then use **Start self-learning cycle** in the dashboard, or:
```powershell
python -m self_learning.cycle --max-pseudo 2000
```

Training is compute-dependent. CPU training can be slow. Use a CUDA-enabled PyTorch build on a supported NVIDIA GPU if available. The app does not claim a model is trained until a checkpoint and metrics exist.

## Dataset quality and experimental protocol

- The Sansar stream is sampled with a deterministic cap and deduplicated during preparation.
- UD Vedic is the supervised morphology source. It is small but manually validated; treat it as a high-quality, narrow-domain benchmark, not broad modern Sanskrit coverage.
- Saamayik provides contemporary prose parallel data and a separate OOD test split. Keep its test/OOD split untouched for evaluation.
- Split monolingual text by document/record before training where possible. Avoid random sentence-level leakage across the same source.
- Report corpus counts, license/source mix, MLM loss, morphology macro-F1, pseudo-label acceptance rate, and pre/post-retention scores. Do not claim translation capability until a decoder has been trained and evaluated on held-out Saamayik test/OOD data.

## Project structure

```text
VYAKARANA-X/
├── backend/              FastAPI inference and training endpoints
├── frontend/             React + Vite + Tailwind dashboard
├── configs/              Model/training settings
├── data/                 Downloaded and prepared corpora
├── data_utils/            Sanskrit tokenizer and vocabulary utilities
├── grammar/               Grammar-aware candidate checks
├── models/                Compact Transformer + morphology head
├── self_learning/         Pseudo-labeling and guarded continual update
├── scripts/               Download and preparation scripts
├── checkpoints/           Versioned model weights and tokenizer
├── train.py                Initial supervised/self-supervised training
└── START_WINDOWS.bat       One-click Windows launcher
```

## Important limitations

- This is a real, trainable research prototype, not a finished Sanskrit translator.
- Heuristic grammar checks cover only selected endings and cannot replace a full Paninian grammar implementation.
- The first model learns MLM representations and UD morphology; translation decoder, full sandhi/samāsa analysis, and a complete rule engine are future research stages.
- Self-training is guarded: pseudo-labels are filtered, and candidate weights are promoted only when held-out morphology performance does not regress. It is not safe to learn from every user query automatically.
- Never use the same examples for pseudo-labelling and final evaluation.

## Citations

- Sansar Sanskrit Corpus, Muse Mesh: https://huggingface.co/datasets/MuseMesh/sansar-sanskrit-corpus
- UD Sanskrit Vedic: https://universaldependencies.org/treebanks/sa_vedic/index.html
- Universal Dependencies licensing: https://universaldependencies.org/contributing/licensing.html
- Sāmayik paper: https://aclanthology.org/2024.lrec-main.1245/
- Sāmayik dataset card: https://huggingface.co/datasets/acomquest/Saamayik
