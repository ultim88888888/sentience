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
