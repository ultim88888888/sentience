"""TDD tests for doppelganger.llm."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from doppelganger.llm import run_claude, CLAUDE_MODEL, CLAUDE_EFFORT


def test_run_claude_invokes_cli_with_stdin(tmp_path):
    fake = MagicMock(returncode=0, stdout="## Bio Lens\nok\n", stderr="")
    with patch("doppelganger.llm.subprocess.run", return_value=fake) as mrun:
        out = run_claude("SYS", "USERBODY", workdir=tmp_path)
    assert out == "## Bio Lens\nok"
    args, kwargs = mrun.call_args
    cmd = args[0]
    assert cmd[0] == "claude" and "-p" in cmd
    assert cmd[cmd.index("--model") + 1] == CLAUDE_MODEL
    assert cmd[cmd.index("--effort") + 1] == CLAUDE_EFFORT
    assert "--no-session-persistence" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert kwargs["input"] == "USERBODY"
    assert str(kwargs["cwd"]) == str(tmp_path)


def test_run_claude_raises_on_nonzero():
    fake = MagicMock(returncode=2, stdout="", stderr="boom")
    with patch("doppelganger.llm.subprocess.run", return_value=fake):
        try:
            run_claude("SYS", "U")
            assert False, "should have raised"
        except RuntimeError as e:
            assert "boom" in str(e)
