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
    assert view["risk_regime"] == {"stance": "no_view"}
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


def test_respond_ablate_memory_uses_empty_memory_and_separate_dir(tmp_path):
    captured = {}

    def fake_run(system, user):
        captured["user"] = user
        return '{"sectors_excited":[{"name":"X","provenance":"extrapolated","citations":[]}]}'

    with patch("doppelganger.respond.run_claude", side_effect=fake_run):
        view = respond("testy-mctest", date(2022, 12, 31), out_dir=tmp_path,
                       soul_path=_soul(tmp_path), ablate_memory=True)
    # written to the ablation dir, NOT the normal views dir
    assert (tmp_path / "testy-mctest" / "views_ablation" / "2022-12-31.json").exists()
    assert not (tmp_path / "testy-mctest" / "views" / "2022-12-31.json").exists()
    # the record section is present but EMPTY (no memory fed)
    assert "YOUR RECORD" in captured["user"]
    between = captured["user"].split("YOUR RECORD", 1)[1].split("# QUESTION", 1)[0]
    assert between.strip().splitlines()[1:] == []   # nothing between the header line and the question
    assert view["sectors_excited"][0]["provenance"] == "extrapolated"
