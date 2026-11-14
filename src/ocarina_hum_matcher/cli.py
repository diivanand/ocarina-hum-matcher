"""The `ocarina` command: listen, identify, play, songs, synth, devices, slides."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import __version__
from .audio import AudioUnavailableError, input_devices, play, record
from .recognizer import (
    DEFAULT_CONFIG,
    DEFAULT_TOLERANCE,
    Analysis,
    FloatArray,
    SongMatch,
    analyze,
    hz_to_midi,
    load_wav,
    match_song,
    rank_songs,
    track_pitch,
)
from .songs import OCARINA_SONGS, describe_shift, find_song, note_name
from .synth import SAMPLE_RATE, render, write_wav


def _utf8() -> bool:
    return (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "") == "utf8"


class LiveView:
    """Draws a level meter and the current note while recording.

    Every half second it re-reads the notes so far, and once a song matches it
    ends the recording, the way the game reacts as soon as you finish a tune.
    """

    def __init__(self, tolerance: float, auto_stop: bool) -> None:
        self.tolerance = tolerance
        self.auto_stop = auto_stop
        self.next_check = 0.5
        self.notes: list[int] = []
        self.match: SongMatch | None = None
        self.tty = sys.stdout.isatty()
        self.fancy = _utf8()

    def __call__(self, signal: FloatArray, rate: int, elapsed: float) -> bool:
        if elapsed >= self.next_check:
            self.next_check = elapsed + 0.5
            self.notes = analyze(signal, rate).midi_notes
            self.match = match_song(self.notes, self.tolerance)
        if self.tty:
            self._draw(signal, rate, elapsed)
        return self.auto_stop and self.match is not None

    def _draw(self, signal: FloatArray, rate: int, elapsed: float) -> None:
        frame = DEFAULT_CONFIG.frame_length(rate)
        tail = signal[-frame:]
        db = 20 * np.log10(float(np.sqrt(np.mean(tail**2))) + 1e-9) if tail.size else -99.0
        level = int(np.clip((db + 60) / 6, 0, 10))
        full, empty = ("█", "·") if self.fancy else ("#", ".")
        pitch = "  --     "
        if len(tail) == frame:
            f0 = track_pitch(tail, rate, replace(DEFAULT_CONFIG, silence_rms=0.003))
            midi = hz_to_midi(f0)
            if midi.size and not np.isnan(midi[0]):
                cents = 100 * (midi[0] - round(midi[0]))
                pitch = f"{note_name(midi[0]):>4} {cents:+3.0f}{'¢' if self.fancy else 'c'}"
        notes = " ".join(note_name(m) for m in self.notes[-10:])
        sys.stdout.write(
            f"\r\x1b[2K  {elapsed:4.1f}s [{full * level}{empty * (10 - level)}] {db:4.0f} dB"
            f"  {pitch}   notes: {notes}"
        )
        sys.stdout.flush()


def _report(analysis: Analysis, ranking: Sequence[SongMatch], tolerance: float) -> SongMatch | None:
    names = " ".join(note_name(m) for m in analysis.midi_notes) or "none"
    tuning = f"tuning {round(analysis.detune * 100) or 0:+d} cents"
    print(f"Detected {len(analysis.notes)} notes: {names}   ({tuning})")
    if not ranking:
        print(
            "Too few notes: the songs have 5 to 8. Hum a bit louder and slower, "
            "with a short gap between repeated notes."
        )
        return None
    best = ranking[0]
    matched = best.error <= tolerance
    if matched:
        print(
            f"\nMatched: {best.name}  (avg error {best.error:.2f} semitones; "
            f"you sang {describe_shift(best.transpose)})"
        )
    else:
        print(f"\nNo match within {tolerance}. Closest: {best.name} at {best.error:.2f} semitones.")
    print("\nClosest songs:")
    for m in ranking[:3]:
        print(f"  {m.name:<20} {m.error:5.2f}{'  <- match' if matched and m is best else ''}")
    return best if matched else None


def _play_song(name: str) -> None:
    print(f"Playing {name} on the ocarina...")
    try:
        play(render(OCARINA_SONGS[name]), SAMPLE_RATE)
    except AudioUnavailableError as error:
        print(f"Can't play audio: {error}", file=sys.stderr)


def _finish(
    args: argparse.Namespace,
    signal: FloatArray,
    rate: int,
    analysis: Analysis,
    ranking: Sequence[SongMatch],
    match: SongMatch | None,
) -> int:
    if getattr(args, "save", None):
        write_wav(args.save, signal, rate)
        print(f"Saved the recording to {args.save}")
    if args.plot or args.show:
        from .plot import save_report, show_report

        if args.plot:
            save_report(args.plot, signal, rate, analysis, ranking, tolerance=args.tolerance)
            print(f"Saved the report to {args.plot}")
    if match and args.play:
        _play_song(match.name)
    if args.show:
        show_report(signal, rate, analysis, ranking, tolerance=args.tolerance)
    return 0 if match else 1


def cmd_listen(args: argparse.Namespace) -> int:
    view = LiveView(args.tolerance, auto_stop=not args.full)
    print(f"Listening for up to {args.seconds:g} s. Hum, sing or whistle a song (Ctrl+C stops).")
    try:
        signal, rate = record(
            args.seconds, device=args.device, sample_rate=args.rate, on_update=view
        )
    except AudioUnavailableError as error:
        print(error, file=sys.stderr)
        return 2
    if view.tty:
        print()
    if not signal.size:
        print("No audio was captured.")
        return 1
    print(f"Recorded {len(signal) / rate:.1f} s at {rate} Hz.\n")
    analysis = analyze(signal, rate)
    ranking = rank_songs(analysis.midi_notes)
    return _finish(
        args, signal, rate, analysis, ranking, _report(analysis, ranking, args.tolerance)
    )


def cmd_identify(args: argparse.Namespace) -> int:
    signal, rate = load_wav(args.wav)
    analysis = analyze(signal, rate)
    ranking = rank_songs(analysis.midi_notes)
    return _finish(
        args, signal, rate, analysis, ranking, _report(analysis, ranking, args.tolerance)
    )


def cmd_play(args: argparse.Namespace) -> int:
    name = find_song(args.song)
    print(f"{name}: {' '.join(note_name(m) for m in OCARINA_SONGS[name])}")
    if args.save:
        write_wav(args.save, render(OCARINA_SONGS[name]))
        print(f"Saved to {args.save}")
    else:
        _play_song(name)
    return 0


def cmd_songs(args: argparse.Namespace) -> int:
    for i, (name, melody) in enumerate(OCARINA_SONGS.items()):
        if i in (0, 6):
            print("Regular songs" if i == 0 else "\nWarp (temple) songs")
        print(f"  {name:<20} {' '.join(note_name(m) for m in melody)}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    name = find_song(args.song)
    audio = render(
        OCARINA_SONGS[name], timbre=args.timbre, transpose=args.transpose, noise=args.noise
    )
    write_wav(args.out, audio)
    print(f"Wrote {name} ({args.timbre}, transposed {args.transpose:+g} semitones) to {args.out}")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    try:
        devices = input_devices()
    except AudioUnavailableError as error:
        print(error, file=sys.stderr)
        return 2
    for index, name, rate, default in devices:
        print(f"{'*' if default else ' '} {index:>3}  {name}  ({rate} Hz)")
    return 0


def cmd_slides(args: argparse.Namespace) -> int:
    import http.server
    import webbrowser
    from importlib.resources import files

    page = (files("ocarina_hum_matcher") / "slides" / "index.html").read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Slides at {url}  (served locally, so the mic demo works). Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocarina",
        description="Recognize Ocarina of Time songs from humming, singing or whistling.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    report = argparse.ArgumentParser(add_help=False)
    report.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="max mean semitone error for a match (default %(default)s)",
    )
    report.add_argument("--plot", type=Path, metavar="PNG", help="save a visual report")
    report.add_argument("--show", action="store_true", help="open the report in a window")

    p = sub.add_parser("listen", parents=[report], help="record from the microphone and match")
    p.add_argument("-s", "--seconds", type=float, default=12.0, help="max recording time")
    p.add_argument("-d", "--device", type=_device, help="input device index or name")
    p.add_argument("--rate", type=int, help="sample rate (default: the device's)")
    p.add_argument("--full", action="store_true", help="keep recording after a match")
    p.add_argument("--save", type=Path, metavar="WAV", help="save the recording")
    p.add_argument(
        "--play",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="play the matched song back (default: on)",
    )
    p.set_defaults(func=cmd_listen)

    p = sub.add_parser("identify", parents=[report], help="match a WAV recording")
    p.add_argument("wav", type=Path)
    p.add_argument(
        "--play",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="play the matched song (default: off)",
    )
    p.set_defaults(func=cmd_identify)

    p = sub.add_parser("play", help="play a song on the ocarina")
    p.add_argument("song", help="name or fragment, e.g. 'storms'")
    p.add_argument("--save", type=Path, metavar="WAV", help="write a WAV instead of playing")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("songs", help="list the songs and their notes")
    p.set_defaults(func=cmd_songs)

    p = sub.add_parser("synth", help="write a WAV of a song, e.g. a test hum")
    p.add_argument("song")
    p.add_argument("out", type=Path)
    p.add_argument("--timbre", choices=("hum", "ocarina"), default="hum")
    p.add_argument("--transpose", type=float, default=-10.0, help="semitones (default -10)")
    p.add_argument("--noise", type=float, default=0.003, help="noise amplitude")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("devices", help="list microphones")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser(
        "slides", help="open the slide deck (20-minute talk + 10-minute demo) in your browser"
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_slides)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except KeyError as error:
        print(error.args[0], file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # output piped into e.g. `head`, which closed early
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    return code
