"""
Pāṇinian-inspired surface heuristics for Sanskrit morphology.
Conservative: only high-confidence suffix rules are included.
This is NOT a complete Ashtadhyayi implementation.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# Ending tables — keyed by grammatical category label
# ---------------------------------------------------------------------------
ENDINGS = {
    # Nominal cases (selected high-confidence endings)
    "NOM.sg":   ["\u0903", "\u092e\u094d"],           # ः  म्
    "NOM.du":   ["\u094c"],                            # औ
    "NOM.pl":   ["\u0906\u0903"],                      # आः
    "ACC.sg":   ["\u092e\u094d"],                      # म्
    "ACC.pl":   ["\u093e\u0928\u094d"],                # ान्
    "INST.sg":  ["\u090f\u0928"],                      # एन
    "INST.du":  ["\u093e\u092d\u094d\u092f\u093e\u092e\u094d"],  # ाभ्याम्
    "DAT.sg":   ["\u093e\u092f"],                      # ाय
    "DAT.du":   ["\u090f"],                            # े
    "GEN.sg":   ["\u0938\u094d\u092f"],                # स्य
    "GEN.du":   ["\u094b\u0903"],                      # ोः
    "GEN.pl":   ["\u093e\u0923\u093e\u092e\u094d"],   # ाणाम्
    "LOC.sg":   ["\u093f"],                            # ि
    "LOC.pl":   ["\u0937\u0941", "\u090f\u0937\u0941"],# षु  एषु
    "ABL.sg":   ["\u093e\u0924\u094d"],                # ात्
    "ABL.pl":   ["\u092d\u094d\u092f\u0903"],          # भ्यः
    # Verbal endings (present indicative, Parasmaipada)
    "VERB.3sg": ["\u0924\u093f"],                      # ति
    "VERB.3pl": ["\u0928\u094d\u0924\u093f"],          # न्ति
    "VERB.2sg": ["\u0938\u093f"],                      # सि
    "VERB.2pl": ["\u0925"],                            # थ
    "VERB.1sg": ["\u092e\u093f", "\u093e\u092e\u093f"],# मि  ामि
    "VERB.1pl": ["\u093e\u0935"],                      # ाव
    "VERB.opt": ["\u0924\u093e\u092e\u094d", "\u090f\u0924"],  # ताम्  ेत
}

# Flat sorted list (longest first) for greedy match
_ALL_ENDINGS = sorted(
    [(label, e) for label, endings in ENDINGS.items() for e in endings],
    key=lambda x: -len(x[1])
)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _match_ending(tok: str):
    """Return (label, ending) for the first matching suffix, else None."""
    tok = _nfc(tok)
    for label, ending in _ALL_ENDINGS:
        if tok.endswith(_nfc(ending)):
            return label, ending
    return None, None


def inspect_tokens(tokens):
    """
    Analyse each token for surface morphological endings.
    Returns a list of dicts, one per Devanagari token.
    """
    out = []
    for tok in tokens:
        tok = _nfc(str(tok)).strip()
        if not tok or not re.search(r"[\u0900-\u097F]", tok):
            continue
        label, matched = _match_ending(tok)
        if label:
            out.append({
                "token":        tok,
                "surface_hint": label,
                "ending":       matched,
                "rule_note":    f"Recognised ending \u2018{matched}\u2019 \u2192 likely {label}",
                "grammar_score": 0.90,
            })
        else:
            out.append({
                "token":        tok,
                "surface_hint": "unresolved",
                "ending":       None,
                "rule_note":    "No configured suffix rule — retain ambiguity",
                "grammar_score": 0.50,
            })
    return out


def consistency_score(tokens):
    """Return (mean_grammar_score float, details list)."""
    items = inspect_tokens(tokens)
    if not items:
        return 0.0, []
    score = sum(x["grammar_score"] for x in items) / len(items)
    return score, items


def sandhi_check(text: str):
    """
    Simple visarga-sandhi detector.
    Returns True if obvious external visarga sandhi is present.
    Pattern: visarga (ः) immediately followed by a consonant character.
    """
    text = _nfc(text)
    return bool(re.search(r"\u0903[\u0915-\u0939]", text))
