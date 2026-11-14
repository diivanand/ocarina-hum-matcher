"""Recognize Ocarina of Time songs from humming, singing or whistling."""

from .recognizer import (
    DEFAULT_CONFIG,
    DEFAULT_TOLERANCE,
    Analysis,
    DetectorConfig,
    NoteEvent,
    SongMatch,
    analyze,
    extract_notes,
    identify_song,
    load_wav,
    match_song,
    rank_songs,
)
from .songs import OCARINA_SCALE, OCARINA_SONGS, find_song, note_name

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_TOLERANCE",
    "OCARINA_SCALE",
    "OCARINA_SONGS",
    "Analysis",
    "DetectorConfig",
    "NoteEvent",
    "SongMatch",
    "analyze",
    "extract_notes",
    "find_song",
    "identify_song",
    "load_wav",
    "match_song",
    "note_name",
    "rank_songs",
]
