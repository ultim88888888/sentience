"""doppelganger.respond — answer a market-view query AS the subject at date T.

Combines the frozen soul card (who he is) + the time-gated memory feed (what he's
said) in one claude -p pass, returning a structured, provenance-tagged JSON view.
Immersive present-tense framing suppresses model-hindsight.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from doppelganger import config
from doppelganger.llm import run_claude
from doppelganger.memory import load_memory

SURVEY_QUERY = (
    "What are your current market views right now? Which sectors and which tokens are you "
    "excited about, which are you concerned about, and what is your risk-on / risk-off posture "
    "— and why? Remember: it's fine to have a view on a sector but not a specific token (or vice "
    "versa), and fine to have no view on something."
)

_SCHEMA_HINT = """{
  "as_of": "<today's date>", "subject": "<your slug>", "abstained": false,
  "sectors_excited":   [{"name": "...", "why": "...", "provenance": "grounded|persisted|extrapolated", "age_note": "", "citations": [{"date": "YYYY-MM-DD", "quote": "verbatim"}]}],
  "sectors_concerned": [], "tokens_excited": [], "tokens_concerned": [],
  "risk_regime": {"stance": "risk_on|risk_off|neutral|no_view", "why": "...", "provenance": "..."},
  "notes": "..."
}"""


def _name_from_soul(soul_md: str, fallback: str) -> str:
    m = re.search(r"(?m)^name:\s*(.+)$", soul_md)
    return m.group(1).strip() if m else fallback


def build_query_prompt(soul_md: str, memory_text: str, subject: str, t0: date,
                       query: str | None = None) -> tuple[str, str]:
    name = _name_from_soul(soul_md, subject)
    system = f"""You ARE {name}. It is {t0.isoformat()}. The future has not happened yet — you \
know only what you know as of today. Reason in the present tense, as yourself, in real time. Use \
ONLY your character description below and your own record of what you've said and seen; do NOT use \
anything you might know about events after today.

When asked for your market views, answer as a SINGLE JSON object with exactly this shape:
{_SCHEMA_HINT}

RULES:
- Each axis is INDEPENDENT and OPTIONAL. Excited about a sector with no specific token is a complete \
answer; a token concern with no broad sector view is complete; any array may be empty — that is \
expected, not a gap. NEVER manufacture an item just to fill a bucket. If you genuinely have no view \
at all, set "abstained": true with empty arrays.
- provenance per item: "grounded" = you have actually said this (put a dated verbatim quote in \
citations); "persisted" = a view you still hold but haven't restated recently (cite it, set age_note \
e.g. "stated 2021-06, not revisited"); "extrapolated" = inferred from how you think, no direct quote \
(no citation). Cite verbatim.
- Output ONLY the JSON object — no prose before or after.

--- WHO YOU ARE (your soul) ---
{soul_md}"""
    q = query or SURVEY_QUERY
    user = (f"# YOUR RECORD — everything you've said and seen, through today ({t0.isoformat()})\n\n"
            f"{memory_text}\n\n# QUESTION\n\n{q}")
    return system, user


_ARRAYS = ["sectors_excited", "sectors_concerned", "tokens_excited", "tokens_concerned"]


def _parse_view(raw: str, subject: str, t0: date) -> dict:
    i, j = raw.find("{"), raw.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"no JSON object in claude output: {raw[:200]!r}")
    data = json.loads(raw[i:j + 1])          # json.JSONDecodeError subclasses ValueError
    for k in _ARRAYS:
        if not isinstance(data.get(k), list):
            data[k] = []
    if not isinstance(data.get("risk_regime"), dict):
        data["risk_regime"] = {"stance": "no_view"}
    data.setdefault("as_of", t0.isoformat())
    data.setdefault("subject", subject)
    data.setdefault("abstained", False)
    data.setdefault("notes", "")
    return data


def respond(slug: str, t0: date, *, query: str | None = None,
            soul_path: Path | None = None, evidence_path: Path | None = None,
            out_dir: Path | None = None, ablate_memory: bool = False) -> dict:
    sp = soul_path or (config.OUT_DIR / slug / "soul.md")
    soul_md = Path(sp).read_text()
    memory_text = "" if ablate_memory else load_memory(slug, t0, evidence_path=evidence_path).text
    system, user = build_query_prompt(soul_md, memory_text, slug, t0, query)
    raw = run_claude(system, user)
    view = _parse_view(raw, slug, t0)

    subdir = "views_ablation" if ablate_memory else "views"
    base = Path(out_dir or config.OUT_DIR) / slug / subdir
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{t0.isoformat()}.json").write_text(json.dumps(view, indent=2))
    return view
