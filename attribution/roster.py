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

    def resolve(self, name: str | None, prefer=None, allow=None) -> Match:
        """Resolve a name to an a16z member. `prefer` (a list of slugs, e.g. the seminar
        host-prior) disambiguates first/last-name ties. External guests -> is_a16z=False.

        `allow` is the set of slugs CORROBORATED as present in this episode (credited
        authors + people already matched by full/last name in the same post). A bare
        FIRST name promotes to a16z only when its slug is in `allow` or `prefer` —
        otherwise an audience member who self-introduces with a first name that
        collides with a roster member ("I'm Jana", "this is Bill") would be falsely
        tagged a16z. Last-name and full-name matches are identifying and need no
        corroboration. `allow=None` => no corroboration available => suppress
        first-name promotion entirely (used by the strong-match pass-1)."""
        prefer = list(prefer or [])
        allow = set(allow or [])
        if not name or not name.strip():
            return Match(None, None, None, False)
        q = name.lower().strip()
        # 1) exact full-name match
        exact = self._df[self._df["name_l"] == q]
        if len(exact) == 1:
            return self._match(exact.iloc[0])
        toks = q.split()
        if len(toks) == 1:
            t = toks[0]
            # last name is identifying -> trust it directly
            hit = self._pick(self._df[self._df["last"] == t], prefer)
            if hit:
                return hit
            # first name only -> promote to a16z ONLY if corroborated present in episode
            fhit = self._pick(self._df[self._df["first"] == t], prefer)
            if fhit:
                if fhit.slug in prefer or fhit.slug in allow:
                    return fhit
                return Match(None, None, None, False)  # uncorroborated collision -> external
            # ambiguous only if it matched multiple somewhere
            if len(self._df[(self._df["last"] == t) | (self._df["first"] == t)]) > 1:
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
