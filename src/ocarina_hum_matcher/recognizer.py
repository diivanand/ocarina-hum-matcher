"""Recognize ocarina songs in audio.

Pipeline:
    audio -> per-frame pitch (windowed FFT + Harmonic Product Spectrum)
          -> discrete notes -> key-independent match against the songbook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy import fft
from scipy.io import wavfile

from .songs import OCARINA_SONGS

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

# Mean semitone error allowed after transposing to the performer's key. Songs of the
# same length are at least 1.5 apart, so at 0.5 no window is within tolerance of two
# of them, while a few semitone slips are forgiven. Across lengths, Requiem is 1.17
# from six notes of Bolero, and exactly one hum is within 0.5 of both: Bolero with its
# seventh note sung as F4, exactly 0.5 from each. Ties go to the longer song (Bolero).
# 1.5 accepted Zelda's Lullaby backwards; 1.0 accepted Epona's Song backwards as
# Saria's Song.
DEFAULT_TOLERANCE: Final = 0.5


@dataclass(frozen=True)
class DetectorConfig:
    frame_seconds: float = 0.04  # 40 ms holds 3+ periods of the lowest pitch
    hop_seconds: float = 0.01
    zero_pad: int = 4  # FFT length multiplier, for finer frequency bins
    min_freq: float = 80.0
    max_freq: float = 1200.0
    harmonics: int = 3  # spectra multiplied together by the HPS
    fundamental_floor: float = 0.1  # fundamental must reach 10% of the loudest bin
    silence_rms: float = 0.02  # relative to the recording's peak
    min_note_seconds: float = 0.08  # shorter runs are slides or wobble
    min_rest_seconds: float = 0.03  # a rest this long separates repeated notes

    def frame_length(self, sample_rate: int) -> int:
        return round(self.frame_seconds * sample_rate)

    def hop_length(self, sample_rate: int) -> int:
        return max(1, round(self.hop_seconds * sample_rate))


DEFAULT_CONFIG: Final = DetectorConfig()


class Span(NamedTuple):
    midi: int
    start: int  # first frame
    stop: int  # one past the last frame


@dataclass(frozen=True)
class NoteEvent:
    midi: int
    start: float  # seconds
    end: float


@dataclass(frozen=True)
class Analysis:
    times: FloatArray  # frame centres, seconds
    frequencies: FloatArray  # fundamental per frame in Hz, NaN where unpitched
    detune: float  # performer's overall offset from A440 tuning, in semitones
    notes: tuple[NoteEvent, ...]

    @property
    def midi_notes(self) -> list[int]:
        return [note.midi for note in self.notes]

    @property
    def midi_track(self) -> FloatArray:
        return hz_to_midi(self.frequencies)


@dataclass(frozen=True)
class SongMatch:
    name: str
    error: float  # mean |semitone error| after the best transposition
    start: int  # index of the first matched note
    transpose: float  # semitones from the ocarina's pitch to the performer's


def load_wav(path: str | Path) -> tuple[FloatArray, int]:
    """Read a WAV file as mono float64."""
    sample_rate, data = wavfile.read(path)
    samples = np.asarray(data, dtype=np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, int(sample_rate)


def normalize(signal: FloatArray) -> FloatArray:
    """Remove DC and scale the peak to 1 (silence is returned unchanged)."""
    centred = signal - signal.mean() if signal.size else signal.copy()
    peak = float(np.abs(centred).max(initial=0.0))
    return centred / peak if peak > 0 else centred


def magnitude_spectrum(
    frames: FloatArray, sample_rate: int, zero_pad: int = DEFAULT_CONFIG.zero_pad
) -> tuple[FloatArray, float]:
    """Hann-windowed, zero-padded magnitude spectrum of each frame (last axis).

    Returns the spectra and the bin spacing in Hz.
    """
    n = frames.shape[-1]
    n_fft = fft.next_fast_len(n * zero_pad, real=True)
    spectrum = np.abs(fft.rfft(frames * np.hanning(n), n=n_fft, axis=-1))
    return spectrum, sample_rate / n_fft


def harmonic_product(spectrum: FloatArray, harmonics: int = DEFAULT_CONFIG.harmonics) -> FloatArray:
    """Harmonic Product Spectrum: bin k times bins 2k, 3k, ... (last axis).

    Only the true fundamental collects energy from every harmonic. Bins are kept
    only while their top harmonic is still below Nyquist.
    """
    n_bins = spectrum.shape[-1] // harmonics
    hps = spectrum[..., :n_bins].copy()
    for h in range(2, harmonics + 1):
        hps *= spectrum[..., : h * n_bins : h]
    return hps


def track_pitch(
    signal: FloatArray, sample_rate: int, config: DetectorConfig = DEFAULT_CONFIG
) -> FloatArray:
    """Return the fundamental frequency (Hz) of each frame, NaN where unpitched."""
    frame, hop = config.frame_length(sample_rate), config.hop_length(sample_rate)
    if len(signal) < frame:
        return np.empty(0)

    frames = sliding_window_view(signal, frame)[::hop]
    voiced = np.sqrt(np.mean(frames**2, axis=1)) >= config.silence_rms
    spectrum, bin_hz = magnitude_spectrum(frames, sample_rate, config.zero_pad)
    hps = harmonic_product(spectrum, config.harmonics)

    n_bins = hps.shape[1]
    freqs = np.arange(n_bins) * bin_hz
    in_range = (freqs >= config.min_freq) & (freqs <= config.max_freq)
    if not in_range.any():
        raise ValueError(f"{sample_rate} Hz sample rate is too low for the pitch range")

    # The fundamental must carry real energy. Without this check a pure tone,
    # such as whistling, lets the HPS settle on a subharmonic.
    loudest = spectrum[:, :n_bins][:, in_range].max(axis=1, keepdims=True)
    strong = spectrum[:, :n_bins] >= config.fundamental_floor * loudest
    bins = np.where(in_range & strong, hps, 0.0).argmax(axis=1)

    f0 = refine_peaks(spectrum, bins, max_steps=2 * config.zero_pad) * bin_hz
    # A peak that climbed out of range (e.g. into the DC skirt of a click) has no pitch.
    pitched = voiced & (f0 >= config.min_freq) & (f0 <= config.max_freq)
    return np.where(pitched, f0, np.nan)


def refine_peaks(spectrum: FloatArray, bins: IntArray, max_steps: int) -> FloatArray:
    """Climb each bin to the top of its spectral lobe, then interpolate between bins.

    A parabola through the log magnitudes of the peak and its two neighbours
    locates the true peak to a fraction of a bin.
    """
    rows = np.arange(len(bins))
    last = spectrum.shape[1] - 2
    for _ in range(max_steps):
        bins = np.clip(bins, 1, last)
        left, here, right = (spectrum[rows, bins + d] for d in (-1, 0, 1))
        step = np.where(right > np.maximum(left, here), 1, np.where(left > here, -1, 0))
        if not step.any():
            break
        bins = bins + step
    bins = np.clip(bins, 1, last)

    left, here, right = (np.log(spectrum[rows, bins + d] + 1e-12) for d in (-1, 0, 1))
    curvature = left - 2 * here + right
    offset = np.divide(
        0.5 * (left - right), curvature, out=np.zeros_like(curvature), where=curvature < 0
    )
    return offset + bins


def hz_to_midi(frequencies: FloatArray) -> FloatArray:
    """MIDI note numbers (fractional); NaN stays NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 69 + 12 * np.log2(frequencies / 440.0)


def estimate_detune(midi: FloatArray) -> float:
    """Performer's overall detune in semitones, in (-0.5, 0.5].

    The circular mean of the fractional MIDI values, so a singer who is
    consistently 45 cents sharp still rounds to the intended notes.
    """
    voiced = midi[~np.isnan(midi)]
    if not voiced.size:
        return 0.0
    return float(np.angle(np.mean(np.exp(2j * np.pi * voiced))) / (2 * np.pi))


def segment_notes(
    track: Sequence[int | None], *, min_note_frames: int, min_rest_frames: int
) -> list[Span]:
    """Collapse a per-frame note track into discrete notes.

    Short runs (slides, wobble) are dropped. A run with the same pitch as the
    previous note counts as a new note only if a rest separates them, which is
    how repeated notes like the A-A in Serenade of Water are kept.
    """
    spans: list[Span] = []
    rested = True
    position = 0
    for value, run in groupby(track):
        length = sum(1 for _ in run)
        if value is None:
            rested = rested or length >= min_rest_frames
        elif length >= min_note_frames:
            if rested or spans[-1].midi != value:
                spans.append(Span(value, position, position + length))
            else:  # the same note resumed after a wobble
                spans[-1] = spans[-1]._replace(stop=position + length)
            rested = False
        position += length
    return spans


def analyze(
    signal: FloatArray, sample_rate: int, config: DetectorConfig = DEFAULT_CONFIG
) -> Analysis:
    """Pitch-track a recording and split it into notes."""
    frequencies = track_pitch(normalize(signal), sample_rate, config)
    hop = config.hop_length(sample_rate) / sample_rate
    centre = config.frame_length(sample_rate) / sample_rate / 2
    times = (np.arange(len(frequencies)) * hop + centre).astype(np.float64)

    midi = hz_to_midi(frequencies)
    detune = estimate_detune(midi)
    track = [None if np.isnan(m) else int(round(m - detune)) for m in midi]
    spans = segment_notes(
        track,
        min_note_frames=max(1, round(config.min_note_seconds / config.hop_seconds)),
        min_rest_frames=max(1, round(config.min_rest_seconds / config.hop_seconds)),
    )
    notes = tuple(
        NoteEvent(s.midi, float(times[s.start] - hop / 2), float(times[s.stop - 1] + hop / 2))
        for s in spans
    )
    return Analysis(times, frequencies, detune, notes)


def extract_notes(
    signal: FloatArray, sample_rate: int, config: DetectorConfig = DEFAULT_CONFIG
) -> list[int]:
    """Convert audio into a list of discrete MIDI note numbers."""
    return analyze(signal, sample_rate, config).midi_notes


def rank_songs(
    notes: Sequence[int], songbook: Mapping[str, Sequence[int]] = OCARINA_SONGS
) -> list[SongMatch]:
    """Score every song against its best-fitting window of `notes`, best first.

    Each window is shifted by its median difference from the song, which is the
    transposition minimizing mean absolute error, so any key works. Ties go to
    the longer song. Songs longer than `notes` are left out.
    """
    played = np.asarray(notes, dtype=np.float64)
    scored: list[tuple[float, int, str, int, float]] = []
    for name, melody in songbook.items():
        if len(played) < len(melody):
            continue
        diffs = sliding_window_view(played, len(melody)) - np.asarray(melody)
        offsets = np.median(diffs, axis=1)
        errors = np.abs(diffs - offsets[:, None]).mean(axis=1)
        best = int(errors.argmin())
        scored.append((float(errors[best]), -len(melody), name, best, float(offsets[best])))
    return [SongMatch(name, error, start, shift) for error, _, name, start, shift in sorted(scored)]


def match_song(
    notes: Sequence[int],
    tolerance: float = DEFAULT_TOLERANCE,
    songbook: Mapping[str, Sequence[int]] = OCARINA_SONGS,
) -> SongMatch | None:
    """Return the best song if its error is within `tolerance` semitones."""
    ranking = rank_songs(notes, songbook)
    return ranking[0] if ranking and ranking[0].error <= tolerance else None


def identify_song(
    path: str | Path,
    tolerance: float = DEFAULT_TOLERANCE,
    config: DetectorConfig = DEFAULT_CONFIG,
) -> SongMatch | None:
    """Detect which ocarina song a WAV recording contains, if any."""
    signal, sample_rate = load_wav(path)
    return match_song(extract_notes(signal, sample_rate, config), tolerance)
