"""TDD tests for doppelganger.soul."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.soul import load_soul_inputs

FIX = Path("tests/fixtures/doppelganger")


def test_load_soul_inputs_gates_by_t0():
    identity, evidence = load_soul_inputs(
        "testy-mctest", date(2022, 8, 31),
        evidence_path=None,  # use the built fixture stream below
        identity_path=FIX / "linkedin" / "testy-1.json",
        team_path=FIX / "team.parquet",
        tracked_people_path=FIX / "tracked_people.yaml",
        twitter_path=FIX / "twitter" / "testy.parquet",
        articles_path=FIX / "articles.parquet",
        podcast_path=FIX / "attributed_transcripts.jsonl",
    )
    # identity is truncated to <= 2022-08-31 (Engineer only; GP/CTO are 2023+/2026)
    assert [e.title for e in identity.experience] == ["Engineer"]
    # evidence is filtered to <= 2022-08-31: the 2022-09-01 quote (id "6") is excluded
    assert (evidence["timestamp"].dt.date <= date(2022, 8, 31)).all()
    assert "6" not in set(evidence["id"])
    # sorted ascending
    assert list(evidence["timestamp"]) == sorted(evidence["timestamp"])


from doppelganger.soul import build_extraction_prompt
from doppelganger.schema import IdentityProfile, Experience


def test_build_extraction_prompt_structure():
    import pandas as pd
    identity = IdentityProfile(
        slug="testy-mctest", name="Testy McTest", headline="Investing.", bio="A GP.",
        current_role="Engineer", experience=[Experience("Engineer", "Beta", None, None, None)],
        education=[], socials={},
    )
    ev = pd.DataFrame([
        {"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
         "source_type": "x_original", "text": "Tokens align incentives.",
         "attribution_confidence": 1.0, "context": None},
    ])
    system, user = build_extraction_prompt(identity, ev)
    # system instructions name the required sections and the citation format
    for section in ["How He Thinks", "What He Believes", "What He Attends To",
                    "Open Contradictions", "How He Talks", "Bio Lens"]:
        assert section in system
    assert "[2022-06-01]" not in system          # the date format is described, not pre-filled
    assert '[<YYYY-MM-DD>]' in system or "YYYY-MM-DD" in system
    # user content carries the identity and every evidence item with its date + text
    assert "Testy McTest" in user and "A GP." in user
    assert "2022-06-01" in user and "Tokens align incentives." in user
    assert "x_original" in user


from unittest.mock import patch, MagicMock
from doppelganger.soul import _run_claude


def test_run_claude_invokes_cli_with_stdin(tmp_path):
    fake = MagicMock(returncode=0, stdout="## Bio Lens\nok\n", stderr="")
    with patch("doppelganger.soul.subprocess.run", return_value=fake) as mrun:
        out = _run_claude("SYS", "USERBODY", workdir=tmp_path)
    assert out == "## Bio Lens\nok"          # stripped
    args, kwargs = mrun.call_args
    cmd = args[0]
    assert cmd[0] == "claude" and "-p" in cmd
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "opus"
    assert "--effort" in cmd and cmd[cmd.index("--effort") + 1] == "max"
    assert "--no-session-persistence" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert kwargs["input"] == "USERBODY"      # big payload via stdin, not argv
    assert str(kwargs["cwd"]) == str(tmp_path)


def test_run_claude_raises_on_nonzero():
    fake = MagicMock(returncode=2, stdout="", stderr="boom")
    with patch("doppelganger.soul.subprocess.run", return_value=fake):
        try:
            _run_claude("SYS", "U")
            assert False, "should have raised"
        except RuntimeError as e:
            assert "boom" in str(e)


from datetime import date as _date
from doppelganger.soul import extract_soul


def test_extract_soul_writes_card_with_frontmatter(tmp_path):
    canned = "## Bio Lens\nQuant lens. [2022-06-01] \"Tokens align incentives.\"\n"
    with patch("doppelganger.soul._run_claude", return_value=canned):
        out = extract_soul(
            "testy-mctest", _date(2022, 8, 31), out_dir=tmp_path,
            identity_path=FIX / "linkedin" / "testy-1.json", team_path=FIX / "team.parquet",
            tracked_people_path=FIX / "tracked_people.yaml",
            twitter_path=FIX / "twitter" / "testy.parquet",
            articles_path=FIX / "articles.parquet",
            podcast_path=FIX / "attributed_transcripts.jsonl",
        )
    text = Path(out).read_text()
    assert out == tmp_path / "testy-mctest" / "soul.md"
    assert text.startswith("---\n")                 # YAML frontmatter
    assert "subject: testy-mctest" in text
    assert "t0: 2022-08-31" in text
    assert "evidence_items:" in text
    assert "## Bio Lens" in text and "Tokens align incentives." in text


def test_run_cli_has_soul_subcommand():
    import doppelganger.run as r
    parser = r.build_parser()
    ns = parser.parse_args(["soul", "--subject", "eddy-lazzarin", "--t0", "2022-12-31"])
    assert ns.cmd == "soul" and ns.subject == "eddy-lazzarin" and ns.t0 == "2022-12-31"
