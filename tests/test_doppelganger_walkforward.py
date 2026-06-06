"""TDD tests for doppelganger.walkforward."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from doppelganger.walkforward import quarter_ends


def test_quarter_ends_inclusive_span():
    out = quarter_ends(date(2022, 12, 31), date(2023, 9, 30))
    assert out == [date(2022, 12, 31), date(2023, 3, 31), date(2023, 6, 30), date(2023, 9, 30)]


def test_quarter_ends_skips_partial_quarters():
    out = quarter_ends(date(2023, 1, 15), date(2023, 7, 1))
    assert out == [date(2023, 3, 31), date(2023, 6, 30)]   # Dec-31-2022 excluded, Sep-30 excluded


import json
import pandas as pd
from unittest.mock import patch, MagicMock
from doppelganger.walkforward import run_walkforward


def _ev(tmp_path):
    p = tmp_path / "ev.parquet"
    pd.DataFrame([{"id": "1", "timestamp": pd.Timestamp("2022-06-01", tz="UTC"),
                   "source_type": "x_original", "text": "Tokens align incentives."}]).to_parquet(p)
    return p


def _canned_view():
    # all-extrapolated (no citations) -> audit finds 0 citations -> leaked 0, checked 0
    return {"as_of": "x", "subject": "s", "abstained": False,
            "sectors_excited": [{"name": "ZK", "provenance": "extrapolated", "citations": []}],
            "sectors_concerned": [], "tokens_excited": [], "tokens_concerned": [],
            "risk_regime": {"stance": "risk_on"}, "notes": ""}


def test_run_walkforward_builds_rows_for_both_variants(tmp_path):
    def fake_respond(slug, t0, *, ablate_memory=False, out_dir=None, **kw):
        sub = "views_ablation" if ablate_memory else "views"
        d = Path(out_dir) / slug / sub
        d.mkdir(parents=True, exist_ok=True)
        v = _canned_view()
        (d / f"{t0.isoformat()}.json").write_text(json.dumps(v))
        return v

    with patch("doppelganger.walkforward.respond", side_effect=fake_respond):
        rows = run_walkforward("s", [date(2022, 12, 31)], out_dir=tmp_path, evidence_path=_ev(tmp_path))
    variants = sorted(r["variant"] for r in rows)
    assert variants == ["ablation", "full"]
    full = next(r for r in rows if r["variant"] == "full")
    assert full["risk"] == "risk_on" and full["n_sectors_excited"] == 1
    assert full["extrapolated"] == 1 and full["leaked"] == 0
    wf = json.loads((tmp_path / "s" / "walkforward.json").read_text())
    assert wf["subject"] == "s" and len(wf["rows"]) == 2


def test_run_walkforward_caches_existing_views(tmp_path):
    vd = tmp_path / "s" / "views"
    vd.mkdir(parents=True, exist_ok=True)
    (vd / "2022-12-31.json").write_text(json.dumps(_canned_view()))

    m = MagicMock(side_effect=lambda slug, t0, *, ablate_memory=False, out_dir=None, **kw: (
        (Path(out_dir) / slug / "views_ablation").mkdir(parents=True, exist_ok=True),
        (Path(out_dir) / slug / "views_ablation" / f"{t0.isoformat()}.json").write_text(json.dumps(_canned_view())),
        _canned_view())[-1])
    with patch("doppelganger.walkforward.respond", m):
        rows = run_walkforward("s", [date(2022, 12, 31)], out_dir=tmp_path, evidence_path=_ev(tmp_path))
    assert m.call_count == 1
    assert m.call_args.kwargs["ablate_memory"] is True
    assert len(rows) == 2


def test_run_walkforward_no_ablate(tmp_path):
    def fake_respond(slug, t0, *, ablate_memory=False, out_dir=None, **kw):
        d = Path(out_dir) / slug / "views"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{t0.isoformat()}.json").write_text(json.dumps(_canned_view()))
        return _canned_view()
    with patch("doppelganger.walkforward.respond", side_effect=fake_respond):
        rows = run_walkforward("s", [date(2022, 12, 31)], ablate=False, out_dir=tmp_path, evidence_path=_ev(tmp_path))
    assert len(rows) == 1 and rows[0]["variant"] == "full"


def test_run_cli_has_walkforward_subcommand():
    import doppelganger.run as r
    ns = r.build_parser().parse_args(["walkforward", "--subject", "eddy-lazzarin"])
    assert ns.cmd == "walkforward" and ns.subject == "eddy-lazzarin"
    assert ns.start == "2022-12-31" and ns.end is None and ns.no_ablate is False
    ns2 = r.build_parser().parse_args(["walkforward", "--subject", "x", "--start", "2023-01-01",
                                       "--end", "2023-12-31", "--no-ablate"])
    assert ns2.start == "2023-01-01" and ns2.end == "2023-12-31" and ns2.no_ablate is True
