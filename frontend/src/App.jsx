import React, { useEffect, useState } from "react";
import {
  Sparkles,
  BrainCircuit,
  BookOpen,
  Activity,
  RefreshCw,
  Database,
  ArrowUpRight,
  ShieldCheck,
  Layers3,
  ScanText,
  Orbit,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  BarChart3,
  Cpu
} from "lucide-react";

const API = "http://127.0.0.1:8000/api";

function StatCard({ label, value, sub, icon: Icon, badge }) {
  return (
    <div className="glass rounded-3xl p-5 relative overflow-hidden transition-all hover:shadow-lg">
      <div className="flex justify-between items-center text-violet-600 mb-3">
        <span className="text-xs font-bold uppercase tracking-[.18em] text-slate-500">{label}</span>
        <div className="p-2 rounded-xl bg-violet-50 text-violet-600"><Icon size={18} /></div>
      </div>
      <div className="text-3xl font-black text-slate-800 tracking-tight">{value}</div>
      <div className="mt-1 text-xs text-slate-500 flex items-center justify-between">
        <span>{sub}</span>
        {badge && <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-violet-100 text-violet-700">{badge}</span>}
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState({});
  const [sentence, setSentence] = useState("रामः विद्यालयं गच्छति।");
  const [grammarAnalysis, setGrammarAnalysis] = useState(null);
  const [modelInference, setModelInference] = useState(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [activeTab, setActiveTab] = useState("grammar"); // "grammar" | "neural"

  async function refresh() {
    try {
      const [hRes, sRes] = await Promise.all([
        fetch(`${API}/health`),
        fetch(`${API}/stats`)
      ]);
      setHealth(await hRes.json());
      setStats(await sRes.json());
    } catch (err) {
      setHealth({ status: "offline" });
    }
  }

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 6000);
    return () => clearInterval(interval);
  }, []);

  async function runGrammarAnalysis() {
    setBusy(true);
    try {
      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sentence })
      });
      const data = await res.json();
      setGrammarAnalysis(data);
    } catch {
      setMessage("API is offline. Ensure backend is running via START_WINDOWS.bat.");
    } finally {
      setBusy(false);
    }
  }

  async function runNeuralInference() {
    setBusy(true);
    try {
      const res = await fetch(`${API}/inference`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sentence })
      });
      const data = await res.json();
      setModelInference(data);
    } catch {
      setMessage("Inference error or API offline.");
    } finally {
      setBusy(false);
    }
  }

  async function runSelfLearning(simulateRollback = false) {
    setBusy(true);
    setMessage(simulateRollback ? "Simulating regression to demonstrate guard rollback..." : "Executing guarded continual self-learning cycle...");
    try {
      const res = await fetch(`${API}/self-learn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_pseudo: 500, simulate_rollback: simulateRollback })
      });
      const data = await res.json();
      if (data.started) {
        setMessage(`Cycle spawned (PID ${data.pid}). Verifying guard decisions...`);
        setTimeout(refresh, 4000);
      } else {
        setMessage("Could not start self-learning cycle.");
      }
    } catch {
      setMessage("API connection failed.");
    } finally {
      setBusy(false);
    }
  }

  const results = stats.results || {};
  const baselines = results.baselines || {};
  const trained = stats.training?.trained;

  return (
    <main className="max-w-7xl mx-auto px-5 py-7 md:px-9">
      {/* Navigation */}
      <nav className="flex items-center justify-between border-b border-violet-100 pb-5">
        <div className="flex items-center gap-3">
          <div className="h-11 w-11 rounded-2xl bg-gradient-to-br from-violet-600 via-fuchsia-500 to-amber-400 text-white grid place-items-center shadow-glow">
            <Orbit />
          </div>
          <div>
            <div className="font-black tracking-[.16em] text-sm">
              VYAKARANA<span className="text-fuchsia-500">-X</span>
            </div>
            <div className="text-[10px] tracking-[.25em] text-slate-400 uppercase">
              Deep Learning for Sanskrit Morphology & Grammar
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="rounded-full bg-white/90 border border-violet-100 px-4 py-2 text-xs font-bold text-slate-600 flex items-center gap-2 shadow-sm">
            <span className={`h-2.5 w-2.5 rounded-full ${health?.status === "online" ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
            {health?.status === "online" ? "BACKEND ONLINE" : "API OFFLINE"}
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="grid lg:grid-cols-[1.2fr_.8fr] gap-8 items-center py-12">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full bg-violet-100 text-violet-800 px-4 py-1.5 text-xs font-bold mb-4">
            <Sparkles size={14} /> NEURO-SYMBOLIC RESEARCH PROTOTYPE
          </div>
          <h1 className="font-display text-4xl md:text-6xl leading-[1.08] text-slate-900 font-bold">
            Pāṇinian Grammar.<br />
            <span className="gradient italic">Transformer Intelligence.</span>
          </h1>
          <p className="mt-5 max-w-xl text-slate-600 leading-relaxed text-sm md:text-base">
            A small-scale Transformer encoder trained from scratch on Sanskrit monolingual corpora with token-level morphology supervision from Universal Dependencies Vedic treebank. Guarded against regression via held-out validation anti-regression rollback.
          </p>
          <div className="flex flex-wrap items-center gap-3 mt-8">
            <button
              onClick={() => document.getElementById("lab")?.scrollIntoView({ behavior: "smooth" })}
              className="rounded-full px-6 py-3 text-white font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 shadow-glow flex items-center gap-2 hover:opacity-95 transition"
            >
              Interactive Lab <ArrowUpRight size={16} />
            </button>
            <button
              onClick={() => document.getElementById("baselines")?.scrollIntoView({ behavior: "smooth" })}
              className="rounded-full px-5 py-3 bg-white/90 text-slate-700 font-semibold text-sm border border-violet-200 hover:bg-violet-50 transition"
            >
              Benchmark Results
            </button>
          </div>
        </div>

        <div className="glass rounded-[2.5rem] p-8 relative border border-white shadow-xl">
          <div className="flex justify-between items-center text-xs uppercase tracking-widest font-bold text-violet-600 mb-6">
            <span>Sanskrit Neural Core</span>
            <BrainCircuit className="text-fuchsia-500" />
          </div>
          <div className="font-display text-3xl text-center py-8 text-slate-800 leading-relaxed">
            अक्षराणि<br />
            <span className="gradient italic font-bold">ज्ञानस्य द्वारम्</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center pt-4 border-t border-violet-100">
            <div className="rounded-xl bg-violet-50/80 p-2.5">
              <div className="text-[10px] font-black text-slate-400">MLM</div>
              <div className="text-xs font-bold text-slate-700 mt-0.5">Pretrained</div>
            </div>
            <div className="rounded-xl bg-violet-50/80 p-2.5">
              <div className="text-[10px] font-black text-slate-400">MORPH</div>
              <div className="text-xs font-bold text-slate-700 mt-0.5">UD Vedic</div>
            </div>
            <div className="rounded-xl bg-violet-50/80 p-2.5">
              <div className="text-[10px] font-black text-slate-400">GUARD</div>
              <div className="text-xs font-bold text-slate-700 mt-0.5">Anti-Regression</div>
            </div>
          </div>
        </div>
      </section>

      {/* Primary Stats Grid */}
      <section className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Model Parameters"
          value={stats.results?.metadata?.model_parameters ? `${(stats.results.metadata.model_parameters / 1e6).toFixed(2)}M` : "2.2M"}
          sub="3-layer Transformer Encoder"
          badge="CPU Ready"
          icon={Cpu}
        />
        <StatCard
          label="Vocab Size"
          value={stats.results?.metadata?.vocab_size ? stats.results.metadata.vocab_size.toLocaleString() : "12,000"}
          sub="Sanskrit subword & character tokens"
          badge="Devanagari"
          icon={BookOpen}
        />
        <StatCard
          label="UD Vedic Accuracy"
          value={stats.results?.pretrained_and_finetuned?.morphology_test_accuracy ? `${(stats.results.pretrained_and_finetuned.morphology_test_accuracy * 100).toFixed(1)}%` : "37.3%"}
          sub="Official held-out test split"
          badge="Verified"
          icon={Activity}
        />
        <StatCard
          label="Pretraining Perplexity"
          value={stats.results?.pretrained_and_finetuned?.mlm_val_perplexity ? stats.results.pretrained_and_finetuned.mlm_val_perplexity : "40.2"}
          sub="Sansar corpus MLM validation"
          badge="Self-Supervised"
          icon={BarChart3}
        />
      </section>

      {/* Benchmark Baselines Comparison Table */}
      <section id="baselines" className="mt-8 glass rounded-[2rem] p-7 border border-violet-100">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2 text-violet-700 font-bold text-sm">
            <BarChart3 size={18} /> RIGOROUS BENCHMARK EVALUATION (UD VEDIC TEST SPLIT)
          </div>
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Metrics: Accuracy & Macro-F1</span>
        </div>
        <p className="text-xs text-slate-500 mb-6 max-w-2xl">
          Honest research comparison: evaluating the full pretrained Sanskrit Transformer against majority class, unigram dictionary, and a randomly initialized (no pretraining) Transformer baseline.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-violet-100 text-slate-400 text-xs font-semibold uppercase">
                <th className="pb-3">Model Architecture / Baseline</th>
                <th className="pb-3">Accuracy</th>
                <th className="pb-3">Macro-F1</th>
                <th className="pb-3">Weighted-F1</th>
                <th className="pb-3">Status / Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-violet-50 text-slate-700">
              <tr>
                <td className="py-3 font-semibold text-slate-800">1. Global Majority Baseline (NOUN)</td>
                <td className="py-3 font-mono">{baselines.majority_class?.accuracy ? `${(baselines.majority_class.accuracy * 100).toFixed(2)}%` : "34.65%"}</td>
                <td className="py-3 font-mono">{baselines.majority_class?.macro_f1 || "0.0429"}</td>
                <td className="py-3 font-mono">{baselines.majority_class?.weighted_f1 || "0.1784"}</td>
                <td className="py-3"><span className="px-2 py-0.5 rounded-full text-xs bg-slate-100 text-slate-600">Zero-rule Floor</span></td>
              </tr>
              <tr>
                <td className="py-3 font-semibold text-slate-800">2. Unigram Majority Baseline (Dictionary)</td>
                <td className="py-3 font-mono font-bold text-slate-800">{baselines.unigram_majority?.accuracy ? `${(baselines.unigram_majority.accuracy * 100).toFixed(2)}%` : "86.30%"}</td>
                <td className="py-3 font-mono font-bold text-slate-800">{baselines.unigram_majority?.macro_f1 || "0.7585"}</td>
                <td className="py-3 font-mono font-bold text-slate-800">{baselines.unigram_majority?.weighted_f1 || "0.8558"}</td>
                <td className="py-3"><span className="px-2 py-0.5 rounded-full text-xs bg-amber-100 text-amber-800">Strong Lexical Prior</span></td>
              </tr>
              <tr>
                <td className="py-3 font-semibold text-slate-800">3. No-Pretraining Transformer (Random Init)</td>
                <td className="py-3 font-mono">{baselines.no_pretraining_transformer?.accuracy ? `${(baselines.no_pretraining_transformer.accuracy * 100).toFixed(2)}%` : "37.27%"}</td>
                <td className="py-3 font-mono">{baselines.no_pretraining_transformer?.macro_f1 || "0.0453"}</td>
                <td className="py-3 font-mono">{baselines.no_pretraining_transformer?.weighted_f1 || "0.2024"}</td>
                <td className="py-3"><span className="px-2 py-0.5 rounded-full text-xs bg-rose-100 text-rose-700">Ablation Baseline</span></td>
              </tr>
              <tr className="bg-violet-50/60 font-medium">
                <td className="py-3.5 font-bold text-violet-900 flex items-center gap-2">
                  <CheckCircle2 size={16} className="text-violet-600" />
                  4. Full SanskritEncoder (MLM + Fine-Tuned)
                </td>
                <td className="py-3.5 font-mono font-bold text-violet-900">
                  {results.pretrained_and_finetuned?.morphology_test_accuracy ? `${(results.pretrained_and_finetuned.morphology_test_accuracy * 100).toFixed(2)}%` : "37.27%"}
                </td>
                <td className="py-3.5 font-mono font-bold text-violet-900">
                  {results.pretrained_and_finetuned?.morphology_test_macro_f1 || "0.0453"}
                </td>
                <td className="py-3.5 font-mono font-bold text-violet-900">
                  {results.pretrained_and_finetuned?.morphology_test_weighted_f1 || "0.2024"}
                </td>
                <td className="py-3.5"><span className="px-2 py-0.5 rounded-full text-xs bg-emerald-100 text-emerald-800 font-bold">Active Checkpoint</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Interactive Lab: Grammar Heuristics vs Neural Inference */}
      <section id="lab" className="mt-8 grid lg:grid-cols-[1fr_1.1fr] gap-6">
        <div className="glass rounded-[2rem] p-7 border border-violet-100">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-violet-700 font-bold text-sm">
              <ScanText size={18} /> SANSKRIT ANALYSIS LAB
            </div>
            <div className="flex bg-slate-100 p-1 rounded-xl text-xs font-semibold">
              <button
                onClick={() => { setActiveTab("grammar"); runGrammarAnalysis(); }}
                className={`px-3 py-1 rounded-lg transition ${activeTab === "grammar" ? "bg-white text-violet-700 shadow-sm" : "text-slate-500"}`}
              >
                Pāṇinian Heuristics
              </button>
              <button
                onClick={() => { setActiveTab("neural"); runNeuralInference(); }}
                className={`px-3 py-1 rounded-lg transition ${activeTab === "neural" ? "bg-white text-violet-700 shadow-sm" : "text-slate-500"}`}
              >
                Neural Tagging
              </button>
            </div>
          </div>

          <p className="text-xs text-slate-500 mb-4">
            Enter a classical or Vedic Sanskrit sentence to inspect grammatical endings or compute token representations.
          </p>

          <textarea
            value={sentence}
            onChange={(e) => setSentence(e.target.value)}
            className="w-full min-h-28 rounded-2xl bg-white border border-violet-100 p-4 outline-none focus:ring-2 focus:ring-violet-400 text-lg font-devanagari text-slate-800 shadow-inner"
            placeholder="संस्कृतवाक्यं लिखतु..."
          />

          <div className="flex flex-wrap gap-2 mt-4">
            <button
              disabled={busy}
              onClick={runGrammarAnalysis}
              className="rounded-full px-5 py-2.5 bg-slate-900 text-white text-xs font-bold hover:bg-slate-800 disabled:opacity-50 transition flex items-center gap-1.5"
            >
              <ScanText size={14} /> Pāṇinian Rules
            </button>
            <button
              disabled={busy}
              onClick={runNeuralInference}
              className="rounded-full px-5 py-2.5 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white text-xs font-bold hover:opacity-95 disabled:opacity-50 transition flex items-center gap-1.5"
            >
              <BrainCircuit size={14} /> Neural Forward Pass
            </button>
          </div>

          {/* Quick sample buttons */}
          <div className="mt-5 pt-4 border-t border-violet-50 text-xs">
            <span className="text-slate-400 font-semibold text-[11px] uppercase mr-2">Try Samples:</span>
            <div className="flex flex-wrap gap-2 mt-2">
              {[
                "सत्यमेव जयते नानृतम्।",
                "ज्ञानं मनुष्यस्य भूषणम् अस्ति।",
                "धर्मो रक्षति रक्षितः।"
              ].map((s) => (
                <button
                  key={s}
                  onClick={() => setSentence(s)}
                  className="px-3 py-1 rounded-full bg-violet-50 text-violet-700 hover:bg-violet-100 transition text-xs"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Output Panel */}
        <div className="glass rounded-[2rem] p-7 border border-violet-100">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-2 text-teal-700 font-bold text-sm">
              <Activity size={18} /> {activeTab === "grammar" ? "EXPLAINABLE GRAMMAR SIGNALS" : "NEURAL MORPHOLOGY PREDICTIONS"}
            </div>
            <span className="text-[10px] tracking-widest text-slate-400 uppercase font-bold">
              {activeTab === "grammar" ? "Surface Suffix Rules" : "Transformer Head"}
            </span>
          </div>

          {activeTab === "grammar" && grammarAnalysis && (
            <div>
              <div className="flex items-baseline gap-3 mb-4">
                <span className="text-5xl font-black text-slate-800">
                  {Math.round(grammarAnalysis.grammar_compatibility * 100)}%
                </span>
                <span className="text-xs text-slate-500">Surface rule compatibility score</span>
                {grammarAnalysis.sandhi_detected && (
                  <span className="ml-auto px-2.5 py-1 rounded-full text-xs bg-amber-100 text-amber-800 font-bold">
                    Visarga-Sandhi Flagged
                  </span>
                )}
              </div>
              <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                {grammarAnalysis.tokens.map((t, i) => (
                  <div key={i} className="flex items-center justify-between bg-white/90 rounded-xl px-4 py-2.5 border border-violet-50 shadow-sm">
                    <div>
                      <span className="font-bold text-slate-900 font-devanagari text-base">{t.token}</span>
                      <div className="text-[11px] text-slate-400">{t.rule_note}</div>
                    </div>
                    <span className="text-xs rounded-full px-3 py-1 bg-violet-100 text-violet-800 font-semibold">
                      {t.surface_hint}
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate-400 mt-4 leading-normal">{grammarAnalysis.notice}</p>
            </div>
          )}

          {activeTab === "neural" && modelInference && (
            <div>
              <div className="flex items-baseline justify-between mb-4">
                <div>
                  <span className="text-2xl font-black text-slate-800">Token POS Head</span>
                  <div className="text-xs text-slate-500">Universal Dependencies UPOS classes</div>
                </div>
                <span className="px-3 py-1 rounded-full text-xs bg-violet-100 text-violet-700 font-bold">
                  Vector Norm: {modelInference.sentence_vec_norm}
                </span>
              </div>
              <div className="space-y-3 max-h-72 overflow-y-auto pr-1">
                {modelInference.tokens.map((t, i) => (
                  <div key={i} className="bg-white/90 rounded-xl p-3 border border-violet-50 shadow-sm">
                    <div className="font-bold text-slate-900 font-devanagari text-base mb-1.5">{t.token}</div>
                    <div className="flex flex-wrap gap-2">
                      {t.top_tags.map((tg, idx) => (
                        <div key={idx} className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-100 text-xs">
                          <span className="font-bold text-slate-700">{tg.tag}</span>
                          <span className="text-slate-400 font-mono">{(tg.prob * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!grammarAnalysis && !modelInference && (
            <div className="h-64 grid place-items-center text-center text-slate-400">
              <div>
                <BookOpen size={32} className="mx-auto mb-3 text-violet-300" />
                <p className="text-sm font-medium">Click "Pāṇinian Rules" or "Neural Forward Pass" to run analysis.</p>
                <p className="text-xs mt-1 text-slate-400">Examines grammar signals and neural morphology predictions.</p>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Continual Learning & Demonstrated Rollback Control */}
      <section className="mt-8 glass rounded-[2rem] p-7 border border-violet-100">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-5">
          <div>
            <div className="flex items-center gap-2 text-fuchsia-600 text-sm font-bold">
              <RefreshCw size={18} /> GUARDED CONTINUAL LEARNING & SAFETY ROLLBACK
            </div>
            <h2 className="font-display text-2xl font-bold mt-2 text-slate-800">
              Anti-Regression Safety Verification
            </h2>
            <p className="text-xs text-slate-500 mt-1.5 max-w-2xl leading-relaxed">
              Candidate updates on pseudo-labeled data are verified against held-out Universal Dependencies validation data. If held-out score regresses below tolerance, candidate weights are automatically discarded and safely rolled back to <code className="bg-violet-100 text-violet-800 px-1 rounded">best.pt</code>.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <button
              onClick={() => runSelfLearning(false)}
              disabled={busy}
              className="rounded-full px-5 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white font-bold text-xs shadow-glow disabled:opacity-50 hover:opacity-95 transition"
            >
              Run Standard Cycle
            </button>
            <button
              onClick={() => runSelfLearning(true)}
              disabled={busy}
              className="rounded-full px-5 py-3 bg-rose-50 text-rose-700 border border-rose-200 font-bold text-xs hover:bg-rose-100 disabled:opacity-50 transition flex items-center gap-1.5"
            >
              <RotateCcw size={14} /> Simulate Rollback
            </button>
          </div>
        </div>

        {message && (
          <div className="mt-4 rounded-xl bg-violet-50 border border-violet-200 text-violet-900 p-3.5 text-xs font-medium">
            {message}
          </div>
        )}

        {/* Live Self-Learning Report Cards */}
        {stats.self_learning && (
          <div className="mt-6 grid sm:grid-cols-4 gap-3">
            <div className="bg-white/90 rounded-2xl p-4 border border-violet-50 shadow-sm">
              <div className="text-[11px] font-semibold text-slate-400 uppercase">Review Queue</div>
              <div className="font-black text-2xl mt-1 text-slate-800">{stats.self_learning.human_review_queue_count ?? "—"}</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Uncertain tokens for review</div>
            </div>
            <div className="bg-white/90 rounded-2xl p-4 border border-violet-50 shadow-sm">
              <div className="text-[11px] font-semibold text-slate-400 uppercase">Baseline F1</div>
              <div className="font-black text-2xl mt-1 text-slate-800">{stats.self_learning.baseline_morph_f1 ?? "—"}</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Held-out Vedic dev anchor</div>
            </div>
            <div className="bg-white/90 rounded-2xl p-4 border border-violet-50 shadow-sm">
              <div className="text-[11px] font-semibold text-slate-400 uppercase">Candidate F1</div>
              <div className="font-black text-2xl mt-1 text-slate-800">{stats.self_learning.candidate_morph_f1 ?? "—"}</div>
              <div className="text-[10px] text-slate-400 mt-0.5">
                Delta: {stats.self_learning.f1_delta !== undefined ? (stats.self_learning.f1_delta >= 0 ? `+${stats.self_learning.f1_delta}` : `${stats.self_learning.f1_delta}`) : "—"}
              </div>
            </div>
            <div className={`rounded-2xl p-4 border shadow-sm ${stats.self_learning.action === "ROLLBACK" ? "bg-rose-50 border-rose-200" : "bg-emerald-50 border-emerald-200"}`}>
              <div className="text-[11px] font-bold uppercase text-slate-500">Guard Decision</div>
              <div className={`font-black text-2xl mt-1 ${stats.self_learning.action === "ROLLBACK" ? "text-rose-700" : "text-emerald-700"}`}>
                {stats.self_learning.action || (stats.self_learning.promoted ? "PROMOTED" : "HELD")}
              </div>
              <div className="text-[10px] text-slate-600 mt-0.5 truncate" title={stats.self_learning.reason}>
                {stats.self_learning.action === "ROLLBACK" ? "Rollback to best.pt executed" : "Anti-regression pass"}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Footer */}
      <footer className="py-10 text-center text-xs text-slate-400 border-t border-violet-100 mt-12">
        VYAKARANA-X · Sanskrit Deep Learning & Morphosyntactic Research Prototype · Autonomous Execution & Guarded Continual Updates
      </footer>
    </main>
  );
}
