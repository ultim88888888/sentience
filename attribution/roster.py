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
        self._df["first"] = self._df["name_l"].str.split().str[0]

    @classmethod
    def load(cls) -> "Roster":
        return cls(pd.read_parquet(ROSTER, columns=["slug", "name", "title"]))

    def name_for(self, slug) -> str | None:
        """Canonical display name for a roster slug (or None if not a member)."""
        hit = self._df[self._df["slug"] == slug]
        return hit.iloc[0]["name"] if len(hit) else None

    def _match(self, row) -> Match:
        return Match(row["slug"], row["name"], row["title"], True)

    def _pick(self, hits, prefer) -> Match | None:
        """One hit -> match. Many hits -> disambiguate via the prefer (host-prior) slugs."""
        if len(hits) == 1:
            return self._match(hits.iloc[0])
        if len(hits) > 1 and prefer:
            pref = hits[hits["slug"].isin(prefer)]
            if len(pref) == 1:
                return self._match(pref.iloc[0])
        return None

    def resolve(self, name: str | None, prefer=None) -> Match:
        """Resolve a name to an a16z member. `prefer` (a list of slugs, e.g. the seminar
        host-prior) disambiguates first/last-name ties. External guests -> is_a16z=False."""
        prefer = list(prefer or [])
        if not name or not name.strip():
            return Match(None, None, None, False)
        q = name.lower().strip()
        # 1) exact full-name match
        exact = self._df[self._df["name_l"] == q]
        if len(exact) == 1:
            return self._match(exact.iloc[0])
        toks = q.split()
        if len(toks) == 1:
            # single token: try last name, then first name (hosts are often thanked by first
            # name only — "Thanks, Justin"). Ties broken by the host-prior.
            for col in ("last", "first"):
                hit = self._pick(self._df[self._df[col] == toks[0]], prefer)
                if hit:
                    return hit
            # ambiguous only if it matched multiple somewhere
            if len(self._df[(self._df["last"] == toks[0]) | (self._df["first"] == toks[0])]) > 1:
                return Match(None, None, None, False, ambiguous=True)
            return Match(None, None, None, False)
        # 2) multi-token: match on last name
        bylast = self._df[self._df["last"] == toks[-1]]
        hit = self._pick(bylast, prefer)
        if hit:
            return hit
        if len(bylast) > 1:
            return Match(None, None, None, False, ambiguous=True)
        return Match(None, None, None, False)  # external guest
