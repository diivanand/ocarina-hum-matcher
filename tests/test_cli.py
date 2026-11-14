"""The `ocarina` command, including `listen` against a simulated microphone."""

from __future__ import annotations

import io
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from ocarina_hum_matcher import OCARINA_SONGS, find_song
from ocarina_hum_matcher.audio import AudioUnavailableError
from ocarina_hum_matcher.cli import main
from ocarina_hum_matcher.recognizer import FloatArray
from ocarina_hum_matcher.synth import render


class FakeSoundDevice:
    """Stands in for the sounddevice module: a mic that plays `audio` faster than real time."""

    class PortAudioError(Exception):
        pass

    def __init__(self, audio: FloatArray, rate: int, speed: float = 4.0) -> None:
        self.audio = audio.astype(np.float32)
        self.rate = rate
        self.speed = speed
        self.played: list[tuple[Any, int]] = []
        self.default = SimpleNamespace(device=(0, 0))

    def query_devices(self, device: Any = None, kind: str | None = None) -> Any:
        info = {"name": "Fake mic", "default_samplerate": float(self.rate), "max_input_channels": 1}
        return info if kind else [info]

    def InputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802 (mirrors sounddevice)
        return FakeStream(self, kwargs["callback"])

    def play(self, data: Any, rate: int) -> None:
        self.played.append((data, rate))

    def wait(self) -> None:
        pass


class FakeStream:
    def __init__(self, device: FakeSoundDevice, callback: Callable[..., None]) -> None:
        self.device = device
        self.callback = callback
        self.running = True
        self.thread = threading.Thread(target=self._feed, daemon=True)

    def _feed(self) -> None:
        block, pos = 1024, 0
        audio = self.device.audio
        while self.running:
            chunk = audio[pos : pos + block]
            chunk = np.pad(chunk, (0, block - len(chunk)))  # silence after the song
            self.callback(chunk[:, None], block, None, None)
            pos += block
            time.sleep(block / self.device.rate / self.device.speed)

    def __enter__(self) -> FakeStream:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.running = False
        self.thread.join()


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestCommandLine(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_songs_lists_all_twelve(self) -> None:
        code, out, _ = run("songs")
        self.assertEqual(code, 0)
        for name in OCARINA_SONGS:
            self.assertIn(name, out)

    def test_find_song_accepts_fragments(self) -> None:
        self.assertEqual(find_song("storms"), "Song of Storms")
        self.assertEqual(find_song("zeldas lullaby"), "Zelda's Lullaby")
        with self.assertRaises(KeyError):
            find_song("song")

    def test_synth_then_identify(self) -> None:
        wav = self.dir / "bolero.wav"
        self.assertEqual(run("synth", "bolero", str(wav))[0], 0)
        code, out, _ = run("identify", str(wav))
        self.assertEqual(code, 0)
        self.assertIn("Matched: Bolero of Fire", out)

    def test_identify_saves_report(self) -> None:
        wav, png = self.dir / "minuet.wav", self.dir / "report.png"
        run("synth", "minuet", str(wav), "--transpose", "-13.3")
        code, out, _ = run("identify", str(wav), "--plot", str(png))
        self.assertEqual(code, 0)
        self.assertGreater(png.stat().st_size, 50_000)

    def test_listen_matches_simulated_mic_and_plays_back(self) -> None:
        rate = 48_000
        hum = render(
            OCARINA_SONGS["Song of Storms"],
            timbre="hum",
            transpose=-11.4,
            sample_rate=rate,
            noise=0.003,
        )
        fake = FakeSoundDevice(0.3 * hum, rate)
        started = time.monotonic()
        with mock.patch("ocarina_hum_matcher.audio._sounddevice", return_value=fake):
            code, out, _ = run("listen", "--seconds", "10")
        self.assertEqual(code, 0)
        self.assertIn("Matched: Song of Storms", out)
        self.assertLess(time.monotonic() - started, 6, "should stop as soon as the song matches")
        self.assertEqual(len(fake.played), 1)

    def test_listen_plays_closest_song_when_nothing_matches(self) -> None:
        rate = 48_000
        hum = render(
            OCARINA_SONGS["Song of Storms"], timbre="hum", transpose=-11.4, sample_rate=rate
        )
        fake = FakeSoundDevice(0.3 * hum, rate, speed=8.0)
        with mock.patch("ocarina_hum_matcher.audio._sounddevice", return_value=fake):
            code, out, _ = run("listen", "--seconds", "4", "--tolerance", "-1")
        self.assertEqual(code, 1)
        self.assertIn("No match", out)
        self.assertIn("Playing the closest song, Song of Storms,", out)
        self.assertEqual(len(fake.played), 1)

    def test_listen_explains_missing_portaudio(self) -> None:
        error = AudioUnavailableError("Audio needs the PortAudio library.")
        with mock.patch("ocarina_hum_matcher.audio._sounddevice", side_effect=error):
            code, _, err = run("listen")
        self.assertEqual(code, 2)
        self.assertIn("PortAudio", err)


if __name__ == "__main__":
    unittest.main()
