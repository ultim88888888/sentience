"""Resolve a spoken/inferred name to an a16z roster entry."""
from dataclasses import dataclass
import pandas as pd
from .config import ROSTER

@dataclass
class Match:
    slug: str | None
    name: str | None
    title: str | None
    is_a16z: bool
    ambiguous: bool = False

class Roster:
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._df["name_l"] = self._df["name"].str.lower().str.strip()
        self._df["last"] = self._df["name_l"].str.split().str[-1]

    @classmethod
    def load(cls) -> "Roster":
        return cls(pd.read_parquet(ROSTER, columns=["slug", "name", "title"]))

    def resolve(self, name: str | None) -> Match:
        if not name or not name.strip():
            return Match(None, None, None, False)
        q = name.lower().strip()
        exact = self._df[self._df["name_l"] == q]
        if len(exact) == 1:
            r = exact.iloc[0]
            return Match(r["slug"], r["name"], r["title"], True)
        last = q.split()[-1]
        bylast = self._df[self._df["last"] == last]
        if len(bylast) == 1:
            r = bylast.iloc[0]
            return Match(r["slug"], r["name"], r["title"], True)
        if len(bylast) > 1:
            return Match(None, None, None, False, ambiguous=True)
        return Match(None, None, None, False)  # external guest
