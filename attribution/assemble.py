"""Assemble clean per-person corpora from gated segments.

Each a16z person's corpus = their own kept utterances, grouped by post, with each utterance
optionally preceded by the immediately-prior segment as a `[Q]` context line (so *responses*
stay legible). Context is strictly LOCAL — the prior turn in the same post — never a global
dump of all guest speech. This is what prevents cross-contamination (person A's corpus must
never contain person B's words as if they were A's).
"""

def build_person_conversations(segs, slug: str) -> str:
    """Full labeled dialogues for every post `slug` appears in (has a kept segment).

    Renders EVERY turn of those posts in order as ``Speaker: text`` — both sides of the
    conversation — so a doppelganger can learn how the person *behaves* in dialogue (responds,
    builds, deflects), not just what words they used. Each turn carries the speaker label the
    attributor assigned; the person's own confidently-attributed turns are marked with ⟵.
    """
    df = segs.sort_values(["post_id", "segment_idx"])
    posts = df[(df["is_a16z"] == True) & (df["slug"] == slug)  # noqa: E712
               & (df["kept"] == True)]["post_id"].unique()
    blocks = []
    for pid in posts:
        pdf = df[df["post_id"] == pid]
        lines = [f"=== conversation: post {pid} ==="]
        for _, r in pdf.iterrows():
            who = str(r["speaker"] or "UNKNOWN")
            mine = bool(r["is_a16z"]) and r["slug"] == slug and bool(r["kept"])
            mark = "  ⟵ TARGET" if mine else ""
            lines.append(f"{who}: {str(r['text']).strip()}{mark}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_person_corpus(segs, min_segments: int = 1) -> dict[str, str]:
    df = segs.sort_values(["post_id", "segment_idx"]).reset_index(drop=True)
    kept_a16z = df[(df["is_a16z"] == True) & (df["kept"] == True) & df["slug"].notna()]  # noqa: E712
    counts = kept_a16z.groupby("slug").size()
    slugs = sorted(counts[counts >= min_segments].index)
    out = {}
    for slug in slugs:
        chunks = []
        for _, pdf in df.groupby("post_id"):
            pdf = pdf.reset_index(drop=True)
            mine = pdf.index[(pdf["is_a16z"] == True) & (pdf["slug"] == slug)  # noqa: E712
                             & (pdf["kept"] == True)].tolist()
            if not mine:
                continue
            lines, last = [], -10
            for i in mine:
                # context: the immediately-preceding segment (the prompt this answers),
                # unless it's already this person or already emitted.
                if i - 1 >= 0 and (i - 1) != last:
                    prev = pdf.iloc[i - 1]
                    if not (prev["is_a16z"] and prev["slug"] == slug):
                        lines.append(f"[Q] {str(prev['text']).strip()}")
                lines.append(str(pdf.iloc[i]["text"]).strip())
                last = i
            chunks.append("\n".join(lines))
        out[slug] = "\n\n".join(chunks)
    return out
