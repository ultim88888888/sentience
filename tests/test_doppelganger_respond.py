"""TDD tests for doppelganger.respond."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.respond import build_query_prompt

SOUL = "---\nname: Eddy Lazzarin\nt0: 2022-12-31\n---\n\n## How He Thinks\nMechanism-first."


def test_build_query_prompt_structure():
    system, user = build_query_prompt(SOUL, "[2022-06-01] (x_original) Tokens align incentives.",
                                      "eddy-lazzarin", date(2022, 12, 31))
    # immersive present-tense framing
    assert "You ARE Eddy Lazzarin" in system
    assert "It is 2022-12-31" in system
    assert "future has not happened" in system.lower()
    # schema keys + rules present
    for k in ["sectors_excited", "sectors_concerned", "tokens_excited", "tokens_concerned",
              "risk_regime", "provenance"]:
        assert k in system
    assert "never manufacture" in system.lower()
    assert "abstained" in system
    # soul card embedded in system
    assert "Mechanism-first." in system
    # memory + query in the stdin payload
    assert "Tokens align incentives." in user
    assert "market view" in user.lower()


def test_build_query_prompt_custom_query():
    system, user = build_query_prompt(SOUL, "mem", "eddy-lazzarin", date(2022, 12, 31),
                                      query="What about ZK rollups?")
    assert "What about ZK rollups?" in user


from unittest.mock import patch
import pandas as pd
from doppelganger.respond import respond


def _canned(fenced=False):
    body = ('{"as_of":"2022-12-31","subject":"testy-mctest","abstained":false,'
            '"sectors_excited":[{"name":"DeFi","why":"w","provenance":"grounded","citations":[{"date":"2022-06-01","quote":"q"}]}],'
            '"sectors_concerned":[],"tokens_excited":[],"tokens_concerned":[],'
            '"risk_regime":{"stance":"risk_on","why":"w","provenance":"grounded"},"notes":""}')
    return f"```json\n{body}\n```" if fenced else body


def _soul(tmp_path):
    p = tmp_path / "soul.md"
    p.write_text("---\nname: Testy McTest\n---\n\n## How He Thinks\nMechanism-first.")
    return p


def _ev(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([{"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
                   "source_type": "x_original", "text": "Tokens align incentives.", "context": None}]).to_parquet(p)
    return p


def test_respond_parses_and_writes(tmp_path):
    with patch("doppelganger.respond.run_claude", return_value=_canned(fenced=True)):
        view = respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path,
                       soul_path=_soul(tmp_path), evidence_path=_ev(tmp_path))
    assert view["subject"] == "testy-mctest" and view["abstained"] is False
    assert view["sectors_excited"][0]["name"] == "DeFi"
    assert view["risk_regime"]["stance"] == "risk_on"
    out = tmp_path / "testy-mctest" / "views" / "2022-12-31.json"
    assert out.exists()
    import json as j
    assert j.loads(out.read_text())["subject"] == "testy-mctest"


def test_respond_normalizes_missing_keys(tmp_path):
    minimal = '{"sectors_excited":[]}'   # everything else missing
    with patch("doppelganger.respond.run_claude", return_value=minimal):
        view = respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path,
                       soul_path=_soul(tmp_path), evidence_path=_ev(tmp_path))
    assert view["sectors_concerned"] == [] and view["tokens_excited"] == []
    assert view["risk_regime"] == {"stance": "no_view", "conviction": None}
    assert view["abstained"] is False and view["as_of"] == "2022-12-31"
    assert view["subject"] == "testy-mctest"


def test_respond_raises_on_non_json(tmp_path):
    with patch("doppelganger.respond.run_claude", return_value="I cannot help with that."):
        try:
            respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path,
                    soul_path=_soul(tmp_path), evidence_path=_ev(tmp_path))
            assert False, "should raise"
        except ValueError:
            pass


def test_run_cli_has_respond_subcommand():
    import doppelganger.run as r
    ns = r.build_parser().parse_args(["respond", "--subject", "eddy-lazzarin", "--t0", "2022-12-31"])
    assert ns.cmd == "respond" and ns.subject == "eddy-lazzarin" and ns.t0 == "2022-12-31"
    assert ns.query is None
    ns2 = r.build_parser().parse_args(["respond", "--subject", "x", "--t0", "2022-12-31", "--query", "Q"])
    assert ns2.query == "Q"


def test_respond_ablate_memory_is_soulless(tmp_path):
    # identity.json provides the stub; soul.md must NOT be read in ablation
    (tmp_path / "testy-mctest").mkdir(parents=True, exist_ok=True)
    (tmp_path / "testy-mctest" / "identity.json").write_text(
        '{"name": "Testy McTest", "headline": "Investing in tokens."}')
    captured = {}

    def fake_run(system, user):
        captured["system"] = system
        captured["user"] = user
        return '{"sectors_excited":[{"name":"X","provenance":"extrapolated","citations":[]}]}'

    with patch("doppelganger.respond.run_claude", side_effect=fake_run):
        view = respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path, ablate_memory=True)
    assert (tmp_path / "testy-mctest" / "views_ablation" / "2022-12-31.json").exists()
    assert "Testy McTest" in captured["system"] and "Investing in tokens." in captured["system"]
    assert "How He Thinks" not in captured["system"]      # no soul card
    between = captured["user"].split("YOUR RECORD", 1)[1].split("# QUESTION", 1)[0]
    assert between.strip().splitlines()[1:] == []
    assert view["sectors_excited"][0]["provenance"] == "extrapolated"


from doppelganger.respond import _coerce_conviction


def test_coerce_conviction_clamps_and_handles_bad_input():
    assert _coerce_conviction(85) == 85
    assert _coerce_conviction("90") == 90
    assert _coerce_conviction(73.6) == 74          # rounds
    assert _coerce_conviction(150) == 100          # clamps high
    assert _coerce_conviction(-5) == 0             # clamps low
    assert _coerce_conviction(None) is None        # absent
    assert _coerce_conviction("high") is None      # unparseable -> None, not a crash


def test_respond_prompt_elicits_conviction():
    system, _ = build_query_prompt(SOUL, "mem", "eddy-lazzarin", date(2022, 12, 31))
    assert "conviction" in system.lower()


def test_respond_normalizes_conviction_on_every_item(tmp_path):
    raw = ('{"sectors_excited":[{"name":"DeFi","why":"w","conviction":150,"provenance":"grounded","citations":[]}],'
           '"tokens_concerned":[{"name":"FOO","conviction":"40"}],'
           '"risk_regime":{"stance":"risk_on","conviction":-3}}')
    with patch("doppelganger.respond.run_claude", return_value=raw):
        view = respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path,
                       soul_path=_soul(tmp_path), evidence_path=_ev(tmp_path))
    assert view["sectors_excited"][0]["conviction"] == 100   # clamped
    assert view["tokens_concerned"][0]["conviction"] == 40   # coerced from str
    assert view["risk_regime"]["conviction"] == 0            # clamped low
