"""Render melodies as audio: an ocarina for playback, a hum for testing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

import numpy as np
from scipy.io import wavfile

from .recognizer import FloatArray
from .songs import midi_to_hz

Timbre = Literal["ocarina", "hum"]
SAMPLE_RATE: Final = 44_100

# Relative amplitudes of harmonics 1, 2, 3, ...
_HARMONICS: Final[dict[str, tuple[float, ...]]] = {
    "ocarina": (1.0, 0.12, 0.04),  # an ocarina is almost a pure tone
    "hum": (1.0, 0.5, 0.25, 0.125),  # humming is rich in harmonics
}


def _tone(midi: float, seconds: float, sample_rate: int, timbre: Timbre) -> FloatArray:
    t = np.arange(round(seconds * sample_rate)) / sample_rate
    if timbre == "ocarina":
        # Breathy attack, gentle release, and vibrato that fades in after the attack.
        envelope = np.clip(np.minimum(t / 0.025, (t[-1] - t) / 0.06), 0.0, 1.0)
        vibrato = 0.12 * np.clip((t - 0.12) / 0.15, 0.0, 1.0) * np.sin(2 * np.pi * 5.5 * t)
    else:
        envelope = np.clip(np.minimum(t, t[-1] - t) / 0.01, 0.0, 1.0)
        vibrato = np.zeros_like(t)
    phase = 2 * np.pi * np.cumsum(midi_to_hz(midi) * 2 ** (vibrato / 12)) / sample_rate
    amps = _HARMONICS[timbre]
    tone = sum(a * np.sin((k + 1) * phase) for k, a in enumerate(amps))
    return np.asarray(envelope * tone / sum(amps), dtype=np.float64)


def render(
    midi_notes: Sequence[float],
    *,
    timbre: Timbre = "ocarina",
    transpose: float = 0.0,
    note_seconds: float | Sequence[float] = 0.4,
    rest_seconds: float = 0.08,
    sample_rate: int = SAMPLE_RATE,
    noise: float = 0.0,
    seed: int = 0,
) -> FloatArray:
    """Render notes separated by short rests, with a rest before and after.

    `transpose` shifts every note by that many semitones (fractions detune it).
    `note_seconds` may give one duration per note.
    """
    durations = (
        [note_seconds] * len(midi_notes)
        if isinstance(note_seconds, float | int)
        else list(note_seconds)
    )
    rest = np.zeros(round(rest_seconds * sample_rate))
    pieces = [rest]
    for midi, seconds in zip(midi_notes, durations, strict=True):
        pieces += [0.8 * _tone(midi + transpose, seconds, sample_rate, timbre), rest]
    audio = np.concatenate(pieces)
    if noise:
        audio = audio + np.random.default_rng(seed).normal(0.0, noise, audio.size)
    return audio


def write_wav(path: str | Path, signal: FloatArray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a 16-bit WAV, scaled so the peak sits at 90% of full scale."""
    peak = float(np.abs(signal).max(initial=0.0)) or 1.0
    wavfile.write(path, sample_rate, (0.9 * 32767 * signal / peak).astype(np.int16))
