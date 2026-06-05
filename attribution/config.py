"""Configuration for the attribution pipeline."""
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "a16z_research"
CORPUS = DATA / "articles.parquet"
TRANSCRIPTS = DATA / "transcripts.parquet"
ROSTER = Path(__file__).resolve().parents[1] / "data" / "a16z_team" / "team.parquet"
AUDIO_CACHE = DATA / "audio_cache"

SEGMENTS_OUT = DATA / "attributed_segments.parquet"
PERSONS_DIR = DATA / "persons"
REPORT_OUT = DATA / "attribution_report.md"
TS_CACHE = DATA / "ts_transcripts"  # timestamped whisper re-transcribe (gitignored)

# LLM attribution — claude -p on the Max subscription ($0), Opus for precision.
LLM_MODEL = "opus"          # resolves to claude-opus-4-8
LLM_EFFORT = "max"
CHUNK_CHARS = 24000         # transcript chunk size for the LLM pass
CHUNK_OVERLAP = 1500        # carry-over so a speaker turn is not split blind

# Precision gate
CONF_MIN = 0.7              # min LLM/segment confidence to keep for the clean corpus

# Diarization (pyannote, local)
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN_OP = "op://local/huggingface/credential"  # free HF token, read via op
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

# Known-host prior: where the host is unnamed in their own intro, seed resolution.
HOST_PRIOR = {
    "research-seminar": ["tim-roughgarden", "justin-thaler"],
    "podcast": ["sonal-chokshi"],
}
