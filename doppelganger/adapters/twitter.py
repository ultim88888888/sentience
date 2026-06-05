"""doppelganger.adapters.twitter — X parquet -> EvidenceItems.

Rules (spec §4 ❶): drop retweets; filter low-substance replies; attach quoted-tweet
text as context (flag when the quoted tweet is not in the subject's corpus); reassemble
self-threads (the subject replying to himself) into a single opinion unit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from doppelganger import config
from doppelganger.schema import EvidenceItem

_URL = re.compile(r"https?://\S+")
_LEADING_MENTIONS = re.compile(r"^(?:@\w+\s+)+")
_TYPE_TO_SOURCE = {"original": "x_original", "quote": "x_quote", "reply": "x_reply"}


def _substance(text: str) -> str:
    """Reply text with leading @mentions and URLs stripped, for the noise filter."""
    return _URL.sub("", _LEADING_MENTIONS.sub("", text or "")).strip()


def load_twitter(parquet_path: Path, subject_slug: str) -> list[EvidenceItem]:
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return []
    author_id = df["author_id"].iloc[0]
    ids = set(df["id"])
    by_id = {r["id"]: r for _, r in df.iterrows()}

    # children: subject tweet replying to another subject tweet (self-continuation)
    children: dict[str, list[str]] = {}
    is_child: set[str] = set()
    for _, r in df.iterrows():
        if r["type"] == "retweet":
            continue
        parent = r["in_reply_to_id"]
        if r["in_reply_to_user_id"] == author_id and parent in ids:
            children.setdefault(parent, []).append(r["id"])
            is_child.add(r["id"])

    items: list[EvidenceItem] = []
    for _, r in df.iterrows():
        if r["type"] == "retweet" or r["id"] in is_child:
            continue
        # roots that are replies-to-others with no self-continuation get the noise filter
        is_reply_to_other = r["type"] == "reply" and r["in_reply_to_user_id"] != author_id
        has_thread = r["id"] in children
        if is_reply_to_other and not has_thread and len(_substance(r["text"])) < config.MIN_REPLY_CONTENT_CHARS:
            continue

        # assemble text: root + self-reply descendants in chronological order
        chain = [r["id"]]
        stack = list(children.get(r["id"], []))
        while stack:
            cid = stack.pop(0)
            chain.append(cid)
            stack = list(children.get(cid, [])) + stack
        chain_rows = sorted((by_id[c] for c in chain), key=lambda x: x["created_at"])
        text = "\n\n".join(str(cr["text"]) for cr in chain_rows)

        context, context_missing = None, False
        if r["type"] == "quote":
            q = r["quoted_id"]
            if q in by_id:
                context = str(by_id[q]["text"])
            else:
                context_missing = True

        items.append(EvidenceItem(
            id=str(r["id"]), subject=subject_slug,
            timestamp=r["created_at"].to_pydatetime(),
            source_type=_TYPE_TO_SOURCE[r["type"]], text=text, speaker_slug=subject_slug,
            attribution_confidence=1.0, thread_id=str(r["conversation_id"]),
            context=context, context_missing=context_missing,
            engagement=int(r["like_count"]) if pd.notna(r["like_count"]) else None,
        ))
    return items
