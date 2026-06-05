"""Assemble clean per-person corpora from gated segments."""

def build_person_corpus(segs) -> dict[str, str]:
    """For each a16z slug, concatenate their kept utterances in post+segment order, keeping
    adjacent guest turns as [GUEST] context lines so responses stay legible."""
    df = segs.sort_values(["post_id", "segment_idx"])
    a16z_slugs = sorted({s for s in df.loc[df["is_a16z"] == True, "slug"].dropna()})  # noqa: E712
    out = {}
    for slug in a16z_slugs:
        lines = []
        for _, r in df.iterrows():
            if r["is_a16z"] and r["slug"] == slug and r["kept"]:
                lines.append(r["text"].strip())
            elif not r["is_a16z"] and r["kept"]:
                lines.append(f"[GUEST]: {r['text'].strip()}")
        out[slug] = "\n".join(lines)
    return out
