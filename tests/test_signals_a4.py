import json
from datetime import date
from unittest.mock import patch
from signals.a4 import soul_anchor, member_call_soul, _MAX_MEM_CHARS
from signals.schema import PeriodSignal

def test_soul_anchor_never_future():
    assert soul_anchor(date(2023,6,30)) == date(2022,12,31)
    assert soul_anchor(date(2024,1,1)) == date(2023,12,31)
    assert soul_anchor(date(2022,1,1)) == date(2022,12,31)  # floor

_OUT = json.dumps({"sectors_excited":[{"name":"privacy","why":"core thesis","conviction":85,"horizon":"structural"}],
    "risk_by_horizon":{"short":{"stance":"risk_off","why":"x"},"long":{"stance":"risk_on","why":"y"}}})

def test_member_call_soul_caps_memory_and_parses():
    big = "x" * (_MAX_MEM_CHARS + 50_000)
    class _Mem: text = big
    with patch("signals.a4.ensure_soul", return_value="SOUL"), \
         patch("signals.a4.load_memory", return_value=_Mem()), \
         patch("signals.a4.run_claude", return_value=_OUT) as rc:
        v = member_call_soul(date(2023,6,30), "ali-yahya", "Ali Yahya",
                             {"as_of":"2023-06-30","market":"m","news":"n"})
    # the user payload (2nd positional arg) must be capped, not the full 190k
    user = rc.call_args[0][1]
    assert len(user) < _MAX_MEM_CHARS + 20_000
    assert v.risk_regime.stance == "risk_off" and {i.item for i in v.items} == {"privacy"}
