"""End-to-end tests: synthesize a hummed WAV, then check what the recognizer hears.

Each song gets a positive test (hummed in another key, 40 cents sharp, is
recognized) and a negative test (the same notes backwards are rejected).
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from ocarina_hum_matcher import OCARINA_SONGS, identify_song
from ocarina_hum_matcher.recognizer import FloatArray
from ocarina_hum_matcher.synth import render, write_wav

TRANSPOSE: Final = -9.6  # an octave below the ocarina and 40 cents sharp, like a hummer


def hum(midi_notes: Sequence[int]) -> FloatArray:
    """Hum-like tones (decaying harmonics) with short rests and a little noise."""
    return render(
        midi_notes,
        timbre="hum",
        transpose=TRANSPOSE,
        note_seconds=0.35,
        rest_seconds=0.12,
        noise=0.003,
    )


class TestOcarinaSongs(unittest.TestCase):
    tmp: tempfile.TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def identify(self, midi_notes: Sequence[int]) -> str | None:
        path = Path(self.tmp.name) / f"{self.id()}.wav"
        write_wav(path, hum(midi_notes))
        match = identify_song(path)
        return match.name if match else None

    def assert_recognized(self, song: str) -> None:
        self.assertEqual(self.identify(OCARINA_SONGS[song]), song)

    def assert_backwards_rejected(self, song: str) -> None:
        self.assertIsNone(self.identify(OCARINA_SONGS[song][::-1]))

    def test_zeldas_lullaby_recognized(self) -> None:
        self.assert_recognized("Zelda's Lullaby")

    def test_zeldas_lullaby_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Zelda's Lullaby")

    def test_eponas_song_recognized(self) -> None:
        self.assert_recognized("Epona's Song")

    def test_eponas_song_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Epona's Song")

    def test_sarias_song_recognized(self) -> None:
        self.assert_recognized("Saria's Song")

    def test_sarias_song_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Saria's Song")

    def test_suns_song_recognized(self) -> None:
        self.assert_recognized("Sun's Song")

    def test_suns_song_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Sun's Song")

    def test_song_of_time_recognized(self) -> None:
        self.assert_recognized("Song of Time")

    def test_song_of_time_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Song of Time")

    def test_song_of_storms_recognized(self) -> None:
        self.assert_recognized("Song of Storms")

    def test_song_of_storms_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Song of Storms")

    def test_minuet_of_forest_recognized(self) -> None:
        self.assert_recognized("Minuet of Forest")

    def test_minuet_of_forest_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Minuet of Forest")

    def test_bolero_of_fire_recognized(self) -> None:
        self.assert_recognized("Bolero of Fire")

    def test_bolero_of_fire_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Bolero of Fire")

    def test_serenade_of_water_recognized(self) -> None:
        self.assert_recognized("Serenade of Water")

    def test_serenade_of_water_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Serenade of Water")

    def test_nocturne_of_shadow_recognized(self) -> None:
        self.assert_recognized("Nocturne of Shadow")

    def test_nocturne_of_shadow_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Nocturne of Shadow")

    def test_requiem_of_spirit_recognized(self) -> None:
        self.assert_recognized("Requiem of Spirit")

    def test_requiem_of_spirit_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Requiem of Spirit")

    def test_prelude_of_light_recognized(self) -> None:
        self.assert_recognized("Prelude of Light")

    def test_prelude_of_light_backwards_rejected(self) -> None:
        self.assert_backwards_rejected("Prelude of Light")


if __name__ == "__main__":
    unittest.main()
