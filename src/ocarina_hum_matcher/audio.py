"""Microphone input and speaker output through PortAudio (via sounddevice)."""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .recognizer import FloatArray

# Called about 10 times a second with the audio so far, its sample rate and the
# elapsed seconds. Returning True stops the recording.
UpdateCallback = Callable[[FloatArray, int, float], bool]


class AudioUnavailableError(RuntimeError):
    pass


def _sounddevice() -> Any:
    try:
        import sounddevice
    except (ImportError, OSError) as error:
        raise AudioUnavailableError(
            "Audio needs the PortAudio library. On Debian/Ubuntu run "
            "`sudo apt install libportaudio2`; on macOS and Windows it ships with "
            f"the sounddevice wheel. ({error})"
        ) from error
    return sounddevice


def input_devices() -> list[tuple[int, str, int, bool]]:
    """(index, name, default sample rate, is default) for every input device."""
    sd = _sounddevice()
    default = sd.default.device[0]
    return [
        (i, d["name"], int(d["default_samplerate"]), i == default)
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def record(
    seconds: float,
    *,
    device: int | str | None = None,
    sample_rate: int | None = None,
    on_update: UpdateCallback | None = None,
    update_seconds: float = 0.1,
) -> tuple[FloatArray, int]:
    """Record mono audio until `seconds` pass, `on_update` returns True, or Ctrl+C."""
    sd = _sounddevice()
    rate = int(sample_rate or sd.query_devices(device, "input")["default_samplerate"])
    blocks: queue.Queue[NDArray[np.float32]] = queue.Queue()

    def callback(indata: NDArray[np.float32], frames: int, time_info: Any, status: Any) -> None:
        blocks.put(indata[:, 0].copy())

    chunks: list[NDArray[np.float32]] = []
    try:
        with sd.InputStream(
            device=device, channels=1, samplerate=rate, dtype="float32", callback=callback
        ):
            start = time.monotonic()
            next_update = start
            while (now := time.monotonic()) - start < seconds:
                try:
                    chunks.append(blocks.get(timeout=0.05))
                except queue.Empty:
                    continue
                if on_update and now >= next_update:
                    next_update = now + update_seconds
                    if on_update(np.concatenate(chunks).astype(np.float64), rate, now - start):
                        break
    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as error:
        raise AudioUnavailableError(f"Couldn't open the microphone: {error}") from error
    while not blocks.empty():
        chunks.append(blocks.get_nowait())
    signal = np.concatenate(chunks).astype(np.float64) if chunks else np.zeros(0)
    return signal, rate


def play(signal: FloatArray, sample_rate: int) -> None:
    """Play audio through the default output device and wait until it ends."""
    sd = _sounddevice()
    try:
        sd.play(signal.astype(np.float32), sample_rate)
        sd.wait()
    except sd.PortAudioError as error:
        raise AudioUnavailableError(f"Couldn't open the speakers: {error}") from error
