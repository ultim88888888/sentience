"""The canonical vocabulary: seed sectors + everything minted so far. Pure
bookkeeping — the semantic fit-or-mint judgment is in signals/canonicalize.py."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

from signals import config


@dataclass
class Registry:
    sectors: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    token_parent: dict[str, str] = field(default_factory=dict)

    @classmethod
    def seed(cls) -> "Registry":
        return cls(sectors=list(config.SEED_SECTORS), tokens=[], token_parent={})

    def mint_sector(self, name: str) -> None:
        if name not in self.sectors:
            self.sectors.append(name)

    def mint_token(self, ticker: str, *, parent_sector: str | None = None) -> None:
        if ticker not in self.tokens:
            self.tokens.append(ticker)
        if parent_sector:
            self.token_parent[ticker] = parent_sector

    def to_dict(self) -> dict:
        return {"sectors": self.sectors, "tokens": self.tokens,
                "token_parent": self.token_parent}


def save_registry(reg: Registry, path: Path | None = None) -> None:
    path = path or config.REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg.to_dict(), indent=2))


def load_registry(path: Path | None = None) -> Registry:
    path = path or config.REGISTRY_PATH
    if not Path(path).exists():
        return Registry.seed()
    d = json.loads(Path(path).read_text())
    return Registry(sectors=d.get("sectors", []), tokens=d.get("tokens", []),
                    token_parent=d.get("token_parent", {}))
