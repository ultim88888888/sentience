"""A1 consensus extraction: read the blended as-of-T corpus, emit the a16z house
market view. Free-form sector/token names (no taxonomy — canonicalize.py fits them
later). Recency-privileging (long window contains stale + fresh statements)."""
from __future__ import annotations
import json
from datetime import date

from doppelganger.llm import run_claude
from signals.corpus import assemble_corpus
from signals.schema import Citation, SignalItem, RiskRegime, PeriodSignal

_SYSTEM_TMPL = """You are analyzing the public corpus of the a16z crypto team as of {t}.
It is {t}. The future has not happened yet. Use ONLY the statements provided.

Produce the TEAM'S CONSENSUS market view — what the house is collectively excited about,
concerned about, and its overall risk posture — as of {t}.

CRITICAL — recency: the corpus spans up to {window} months. When statements conflict,
weight the MOST RECENT view. If an old stance was later reversed, report the current
stance and record the reversal in age_note (e.g. "bullish 2022, cooled 2024"). A view
stated once long ago and never restated is `persisted` (cite it, note its age).

Name sectors and tokens in your OWN words — do not force them into any fixed taxonomy.
For tokens, give the sector they belong to in parent_sector.

Every excited/concerned entry needs: name, why, conviction (0-100 intensity, NOT a
probability), horizon ("tactical" | "structural"), provenance ("grounded"=stated with a
dated verbatim quote | "persisted"=held but not recently restated | "extrapolated"=inferred),
age_note (or null), and citations (verbatim quotes <=25 words with ISO dates; required for
grounded/persisted).

Output JSON only:
{{"sectors_excited": [...], "sectors_concerned": [...], "tokens_excited": [...],
  "tokens_concerned": [...],
  "risk_regime": {{"stance": "risk_on|risk_off|neutral|no_view", "conviction": 0-100,
                   "why": "...", "provenance": "..."}},
  "notes": "..."}}"""


def build_a1_prompt(corpus_text: str, t: date, *, window_months: int = 18) -> tuple[str, str]:
    system = _SYSTEM_TMPL.format(t=t.isoformat(), window=window_months)
    return system, corpus_text


def extract_a1(t: date, *, window_months: int, twitter_paths, articles, distillates) -> PeriodSignal:
    corpus = assemble_corpus(t=t, window_months=window_months, twitter_paths=twitter_paths,
                             articles=articles, distillates=distillates)
    system, user = build_a1_prompt(corpus, t, window_months=window_months)
    raw = run_claude(system, user)
    return parse_extraction(raw, t=t)


_ARRAY_STANCE = {"sectors_excited": ("sector", "bullish"),
                 "sectors_concerned": ("sector", "bearish"),
                 "tokens_excited": ("token", "bullish"),
                 "tokens_concerned": ("token", "bearish")}


def parse_extraction(raw: str, t: date) -> PeriodSignal:
    obj = _extract_json(raw)
    items: list[SignalItem] = []
    for key, (item_type, stance) in _ARRAY_STANCE.items():
        for e in obj.get(key, []) or []:
            items.append(SignalItem(
                item=e["name"], item_type=item_type,
                parent_sector=e.get("parent_sector"), stance=stance,
                conviction=e.get("conviction", 50), horizon=e.get("horizon", "tactical"),
                rationale=e.get("why", ""), provenance=e.get("provenance", "extrapolated"),
                age_note=e.get("age_note"),
                citations=tuple(Citation(c["date"], c["quote"]) for c in e.get("citations", []) or []),
            ))
    rr = obj.get("risk_regime") or {"stance": "no_view", "conviction": 0, "why": "", "provenance": "extrapolated"}
    risk = RiskRegime(stance=rr.get("stance", "no_view"), conviction=rr.get("conviction", 0),
                      rationale=rr.get("why", ""), provenance=rr.get("provenance", "extrapolated"))
    return PeriodSignal(as_of=t.isoformat(), approach="A1", items=tuple(items),
                        risk_regime=risk, notes=obj.get("notes", ""))


_MEMBER_SYSTEM_TMPL = """You are analyzing the public statements of {name} (a member of the a16z crypto team) as of {t}.
It is {t}. The future has not happened yet. Use ONLY the statements provided.

Produce {name}'S OWN individual market view — what THEY are personally excited about, concerned
about, and their overall risk posture — as of {t}. This is one person's view, not a team consensus.

CRITICAL — recency: the corpus spans up to {window} months. When statements conflict, weight the
MOST RECENT view. If an old stance was later reversed, report the current stance and record the
reversal in age_note. A view stated once long ago and never restated is `persisted` (cite it, note age).

Name sectors and tokens in your OWN words — do not force them into any fixed taxonomy. For tokens,
give the sector they belong to in parent_sector.

Every excited/concerned entry needs: name, why, conviction (0-100 intensity), horizon
("tactical" | "structural"), provenance ("grounded"|"persisted"|"extrapolated"), age_note (or null),
and citations (verbatim quotes <=25 words with ISO dates; required for grounded/persisted).

Output JSON only:
{{"sectors_excited": [...], "sectors_concerned": [...], "tokens_excited": [...],
  "tokens_concerned": [...],
  "risk_regime": {{"stance": "risk_on|risk_off|neutral|no_view", "conviction": 0-100,
                   "why": "...", "provenance": "..."}},
  "notes": "..."}}"""


def build_member_prompt(corpus_text: str, t, name: str, *, window_months: int = 18):
    from datetime import date
    system = _MEMBER_SYSTEM_TMPL.format(t=t.isoformat(), window=window_months, name=name)
    return system, corpus_text


def extract_member(t, name: str, *, window_months: int, twitter_path, distillates=None):
    from signals.corpus import assemble_corpus
    corpus = assemble_corpus(t=t, window_months=window_months, twitter_paths=[twitter_path],
                             articles=None, distillates=distillates or {})
    system, user = build_member_prompt(corpus, t, name, window_months=window_months)
    raw = run_claude(system, user)
    p = parse_extraction(raw, t=t)
    # tag approach as the member slug-style label for traceability
    return p.__class__(as_of=p.as_of, approach=f"A2a:{name}", items=p.items,
                       risk_regime=p.risk_regime, notes=p.notes)


def _extract_json(raw: str) -> dict:
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return {}
    return json.loads(s[start:end + 1])
