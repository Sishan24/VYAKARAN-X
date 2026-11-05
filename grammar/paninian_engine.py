"""Conservative, explainable suffix checks; not a complete Paninian grammar."""
import re
NOM_ENDINGS = ("ः","म्","ौ","ाः","ेन","ाय","स्य","े","ात्","भ्याम्","ेषु")
def inspect_tokens(tokens):
    out=[]
    for tok in tokens:
        if not tok or not re.search(r"[\u0900-\u097F]", tok): continue
        matched=next((e for e in NOM_ENDINGS if tok.endswith(e)), None)
        out.append({"token":tok,"surface_hint":matched or "unresolved",
                    "rule_note":f"Recognised surface ending '{matched}'" if matched else "No configured suffix rule; retain ambiguity",
                    "grammar_score":0.90 if matched else 0.50})
    return out
def consistency_score(tokens):
    items=inspect_tokens(tokens)
    return sum(x["grammar_score"] for x in items)/max(1,len(items)), items
