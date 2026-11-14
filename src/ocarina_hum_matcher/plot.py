"""A one-page visual report: what the recognizer heard and how it matched."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import fft
from scipy import signal as sps

from .recognizer import (
    DEFAULT_CONFIG,
    DEFAULT_TOLERANCE,
    Analysis,
    DetectorConfig,
    FloatArray,
    SongMatch,
    harmonic_product,
    magnitude_spectrum,
    normalize,
)
from .songs import OCARINA_SONGS, describe_shift, midi_to_hz, note_name
from .synth import render

C: Final = {
    "bg": "#0d1524",
    "panel": "#142035",
    "grid": "#2c3e5e",
    "text": "#e8eef8",
    "muted": "#9cadc6",
    "gold": "#f3b443",
    "sky": "#63c7ff",
    "good": "#63d69e",
    "bad": "#ff8080",
}
FIGSIZE: Final = (14.0, 11.0)


def build_report(
    fig: Figure,
    signal: FloatArray,
    sample_rate: int,
    analysis: Analysis,
    ranking: Sequence[SongMatch],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    config: DetectorConfig = DEFAULT_CONFIG,
) -> None:
    """Draw the five-panel report onto `fig`."""
    with matplotlib.rc_context(
        {"axes.titlecolor": C["text"], "axes.titlesize": 11.5, "axes.titlelocation": "left"}
    ):
        _draw_report(fig, signal, sample_rate, analysis, ranking, tolerance, config)


def _draw_report(
    fig: Figure,
    signal: FloatArray,
    sample_rate: int,
    analysis: Analysis,
    ranking: Sequence[SongMatch],
    tolerance: float,
    config: DetectorConfig,
) -> None:
    x = normalize(signal)
    best = ranking[0] if ranking else None
    fig.set_facecolor(C["bg"])
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.15, 1, 1),
        hspace=0.55,
        wspace=0.2,
        left=0.07,
        right=0.98,
        top=0.92,
        bottom=0.06,
    )
    axes = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
    ]
    for ax in axes:
        _style(ax)

    if best is None:
        headline, color = f"Too few notes ({len(analysis.notes)}): songs have 5 to 8", C["bad"]
    elif best.error <= tolerance:
        headline, color = f"Matched: {best.name}  (avg error {best.error:.2f} semitones)", C["good"]
    else:
        headline, color = f"No match. Closest: {best.name} ({best.error:.2f} semitones)", C["bad"]
    fig.suptitle(headline, color=color, fontsize=16, fontweight="bold", x=0.07, ha="left")

    _spectrogram(axes[0], x, sample_rate, analysis, config)
    _notes_vs_song(axes[1], analysis, best)
    _ranking(axes[2], ranking, tolerance)
    _average_spectrum(axes[3], x, sample_rate, analysis, best)
    _frame_hps(axes[4], x, sample_rate, analysis, config)


def save_report(
    path: str | Path,
    signal: FloatArray,
    sample_rate: int,
    analysis: Analysis,
    ranking: Sequence[SongMatch],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    config: DetectorConfig = DEFAULT_CONFIG,
) -> None:
    fig = Figure(figsize=FIGSIZE)
    build_report(fig, signal, sample_rate, analysis, ranking, tolerance=tolerance, config=config)
    fig.savefig(path, dpi=120, facecolor=C["bg"])


def show_report(
    signal: FloatArray,
    sample_rate: int,
    analysis: Analysis,
    ranking: Sequence[SongMatch],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    config: DetectorConfig = DEFAULT_CONFIG,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=FIGSIZE)
    build_report(fig, signal, sample_rate, analysis, ranking, tolerance=tolerance, config=config)
    plt.show()


def _style(ax: Axes) -> None:
    ax.set_facecolor(C["panel"])
    for spine in ax.spines.values():
        spine.set_color(C["grid"])
    ax.tick_params(colors=C["muted"], labelsize=9)
    ax.xaxis.label.set_color(C["muted"])
    ax.yaxis.label.set_color(C["muted"])
    ax.grid(color=C["grid"], linewidth=0.6, alpha=0.6)


def _legend(ax: Axes, loc: Literal["upper right", "lower left", "best"] = "upper right") -> None:
    ax.legend(loc=loc, fontsize=8.5, facecolor=C["bg"], edgecolor=C["grid"], labelcolor=C["text"])


def _empty(ax: Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=C["muted"])


def _spectrogram(
    ax: Axes, x: FloatArray, sr: int, analysis: Analysis, config: DetectorConfig
) -> None:
    ax.set_title("Spectrogram of your recording, with the detected pitch and notes", loc="left")
    n = 2 * config.frame_length(sr)
    if len(x) < n:
        _empty(ax, "Recording too short")
        return
    hop = config.hop_length(sr)
    f, t, s = sps.spectrogram(
        x,
        fs=sr,
        window="hann",
        nperseg=n,
        noverlap=n - hop,
        nfft=fft.next_fast_len(2 * n),
        mode="magnitude",
    )
    pitched = analysis.frequencies[~np.isnan(analysis.frequencies)]
    top = float(np.clip(4 * pitched.max(), 600, 2000)) if pitched.size else 1500.0
    keep = f <= top
    db = 20 * np.log10(s[keep] / max(float(s.max()), 1e-12) + 1e-12)
    ax.pcolormesh(t, f[keep], db, vmin=-70, vmax=0, cmap="magma", shading="auto", rasterized=True)
    ax.plot(
        analysis.times, analysis.frequencies, ".", color=C["sky"], ms=2.5, label="detected pitch"
    )
    for note in analysis.notes:
        hz = midi_to_hz(note.midi + analysis.detune)
        ax.hlines(hz, note.start, note.end, color=C["gold"], linewidth=2.5)
        ax.text(
            (note.start + note.end) / 2,
            hz + top * 0.03,
            note_name(note.midi),
            color=C["gold"],
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set(xlim=(0, len(x) / sr), ylim=(0, top), xlabel="Time (s)", ylabel="Frequency (Hz)")
    _legend(ax)


def _notes_vs_song(ax: Axes, analysis: Analysis, best: SongMatch | None) -> None:
    if not analysis.notes:
        ax.set_title("Your notes", loc="left")
        _empty(ax, "No notes detected")
        return
    track = analysis.midi_track - analysis.detune
    ax.plot(
        analysis.times, track, ".", color=C["sky"], ms=2.5, label="your pitch (tuning-corrected)"
    )
    for i, note in enumerate(analysis.notes):
        ax.hlines(
            note.midi,
            note.start,
            note.end,
            color=C["gold"],
            linewidth=5,
            label="detected notes" if i == 0 else "_",
        )
    values = [n.midi for n in analysis.notes]
    if best is None:
        ax.set_title("Your notes", loc="left")
    else:
        song = OCARINA_SONGS[best.name]
        events = analysis.notes[best.start : best.start + len(song)]
        for j, (event, midi) in enumerate(zip(events, song, strict=True)):
            y = midi + best.transpose
            values.append(round(y))
            ax.hlines(
                y,
                event.start,
                event.end,
                color=C["good"],
                linewidth=2,
                linestyles="--",
                label=f"{best.name}, in your key" if j == 0 else "_",
            )
        ax.set_title(f"Your notes vs {best.name} ({describe_shift(best.transpose)})", loc="left")
    lo, hi = min(values) - 2, max(values) + 3
    step = 1 if hi - lo <= 14 else 2
    ticks = list(range(lo, hi + 1, step))
    ax.set_yticks(ticks, [note_name(m) for m in ticks])
    ax.set(ylim=(lo, hi), xlabel="Time (s)", ylabel="Note")
    _legend(ax, "best")


def _ranking(ax: Axes, ranking: Sequence[SongMatch], tolerance: float) -> None:
    ax.set_title("Every song's distance from your notes", loc="left")
    errors = {m.name: m.error for m in ranking}
    names = [m.name for m in ranking] + [n for n in OCARINA_SONGS if n not in errors]
    y = np.arange(len(names))[::-1]
    best = ranking[0] if ranking else None
    for yi, name in zip(y, names, strict=True):
        if name not in errors:
            ax.text(0.05, yi, "needs more notes", va="center", color=C["muted"], fontsize=8)
            continue
        error = errors[name]
        is_best = best is not None and name == best.name
        color = (C["good"] if error <= tolerance else C["bad"]) if is_best else C["grid"]
        ax.barh(yi, max(min(error, 4.0), 0.04), color=color, height=0.62)
        ax.text(
            min(error, 4.0) + 0.05, yi, f"{error:.2f}", va="center", color=C["muted"], fontsize=8
        )
    ax.axvline(tolerance, color=C["gold"], linestyle="--", linewidth=1.2)
    ax.text(
        tolerance + 0.05,
        len(names) - 0.4,
        f"tolerance {tolerance}",
        color=C["gold"],
        fontsize=8,
        va="bottom",
    )
    ax.set_yticks(y, names, fontsize=8.5)
    ax.set(xlim=(0, 4.4), ylim=(-0.7, len(names)), xlabel="Mean semitone error (best key)")
    ax.grid(axis="y", visible=False)


def _average_spectrum(
    ax: Axes, x: FloatArray, sr: int, analysis: Analysis, best: SongMatch | None
) -> None:
    ax.set_title("Average spectrum: your recording vs the matched song", loc="left")
    if len(x) < 512:
        _empty(ax, "Recording too short")
        return
    f, p = sps.welch(x, fs=sr, nperseg=min(len(x), 8192))
    ax.plot(
        f[1:],
        10 * np.log10(p[1:] / max(float(p.max()), 1e-20) + 1e-12),
        color=C["sky"],
        linewidth=1.2,
        label="your recording",
    )
    if best is not None:
        song = OCARINA_SONGS[best.name]
        events = analysis.notes[best.start : best.start + len(song)]
        shift = best.transpose + analysis.detune
        reference = render(
            song,
            transpose=shift,
            sample_rate=sr,
            note_seconds=[max(0.1, e.end - e.start) for e in events],
        )
        f2, p2 = sps.welch(reference, fs=sr, nperseg=8192)
        ax.plot(
            f2[1:],
            10 * np.log10(p2[1:] / max(float(p2.max()), 1e-20) + 1e-12),
            color=C["gold"],
            linewidth=1.2,
            label=f"{best.name} on an ocarina, in your key",
        )
        for midi in sorted(set(song)):
            hz = midi_to_hz(midi + shift)
            ax.axvline(hz, color=C["muted"], linestyle=":", linewidth=0.8)
            ax.text(
                hz,
                1.5,
                note_name(midi + best.transpose),
                color=C["muted"],
                fontsize=7.5,
                ha="center",
                va="bottom",
            )
    ax.set_xscale("log")
    ticks = [60, 100, 200, 300, 500, 1000, 2000]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set(
        xlim=(60, 2000),
        ylim=(-70, 6),
        xlabel="Frequency (Hz, log scale)",
        ylabel="Power (dB, relative)",
    )
    _legend(ax, "lower left")


def _frame_hps(
    ax: Axes, x: FloatArray, sr: int, analysis: Analysis, config: DetectorConfig
) -> None:
    if not analysis.notes:
        ax.set_title("How one frame got its pitch", loc="left")
        _empty(ax, "No pitched frames")
        return
    note = max(analysis.notes, key=lambda n: n.end - n.start)
    t = (note.start + note.end) / 2
    frame = config.frame_length(sr)
    start = int(np.clip(round(t * sr) - frame // 2, 0, len(x) - frame))
    spectrum, bin_hz = magnitude_spectrum(x[start : start + frame], sr, config.zero_pad)
    product = np.cbrt(harmonic_product(spectrum, config.harmonics))
    f0 = float(analysis.frequencies[int(np.abs(analysis.times - t).argmin())])
    if np.isnan(f0):
        f0 = float(midi_to_hz(note.midi + analysis.detune))

    top = min(1500.0, 4 * f0 + 150)
    freqs = np.arange(len(spectrum)) * bin_hz
    show = freqs <= top
    hps_freqs = freqs[: len(product)]
    in_range = (hps_freqs >= config.min_freq) & (hps_freqs <= min(top, config.max_freq))
    ax.plot(
        freqs[show],
        spectrum[show] / spectrum[show].max(),
        color=C["sky"],
        linewidth=1.2,
        label="spectrum |X|",
    )
    ax.plot(
        hps_freqs[hps_freqs <= top],
        product[hps_freqs <= top] / product[in_range].max(),
        color=C["gold"],
        linewidth=1.4,
        label="harmonic product (cube root)",
    )
    ax.axvline(f0, color=C["good"], linestyle="--", linewidth=1.2)
    ax.text(
        f0, 1.04, f" {f0:.1f} Hz = {note_name(note.midi)}", color=C["good"], fontsize=9, va="bottom"
    )
    for h in range(2, config.harmonics + 1):
        if h * f0 < top:
            peak = spectrum[round(h * f0 / bin_hz)] / spectrum[show].max()
            ax.text(h * f0, peak + 0.04, f"{h}f₀", color=C["muted"], fontsize=9, ha="center")
    ax.set_title(
        f"How one frame ({t:.2f} s) got its pitch: harmonics line up only at f₀", loc="left"
    )
    ax.set(xlim=(0, top), ylim=(0, 1.15), xlabel="Frequency (Hz)", ylabel="Relative magnitude")
    _legend(ax)
