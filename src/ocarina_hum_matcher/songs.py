"""The ocarina's scale and the 12 songs of Ocarina of Time, as MIDI note numbers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

# The ocarina's five notes.
OCARINA_SCALE: Final[Mapping[str, int]] = {"D4": 62, "F4": 65, "A4": 69, "B4": 71, "D5": 74}

_SHEET_MUSIC: Final[Mapping[str, str]] = {
    # Regular songs
    "Zelda's Lullaby": "B4 D5 A4 B4 D5 A4",
    "Epona's Song": "D5 B4 A4 D5 B4 A4",
    "Saria's Song": "F4 A4 B4 F4 A4 B4",
    "Sun's Song": "A4 F4 D5 A4 F4 D5",
    "Song of Time": "A4 D4 F4 A4 D4 F4",
    "Song of Storms": "D4 F4 D5 D4 F4 D5",
    # Warp (temple) songs
    "Minuet of Forest": "D4 D5 B4 A4 B4 A4",
    "Bolero of Fire": "F4 D4 F4 D4 A4 F4 A4 F4",
    "Serenade of Water": "D4 F4 A4 A4 B4",
    "Nocturne of Shadow": "B4 A4 A4 D4 B4 A4 F4",
    "Requiem of Spirit": "D4 F4 D4 A4 F4 D4",
    "Prelude of Light": "D5 A4 D5 A4 B4 D5",
}

OCARINA_SONGS: Final[Mapping[str, tuple[int, ...]]] = {
    name: tuple(OCARINA_SCALE[note] for note in sheet.split())
    for name, sheet in _SHEET_MUSIC.items()
}

_NOTE_NAMES: Final = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi: float) -> str:
    """Name of the nearest MIDI note, e.g. 69 -> 'A4'."""
    m = round(midi)
    return f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def describe_shift(semitones: float) -> str:
    """'12.0 semitones below the ocarina', or 'in the ocarina's key' when close to 0."""
    if abs(semitones) < 0.25:
        return "in the ocarina's key"
    return f"{abs(semitones):.1f} semitones {'below' if semitones < 0 else 'above'} the ocarina"


def find_song(query: str) -> str:
    """Resolve a song from a case-insensitive name or unique fragment, e.g. 'storms'."""
    wanted = query.casefold().replace("'", "")
    names = {name.casefold().replace("'", ""): name for name in OCARINA_SONGS}
    if wanted in names:
        return names[wanted]
    hits = [name for key, name in names.items() if wanted in key]
    if len(hits) == 1:
        return hits[0]
    options = ", ".join(hits or OCARINA_SONGS)
    raise KeyError(f"{query!r} matches {'several songs' if hits else 'no song'}: {options}")
