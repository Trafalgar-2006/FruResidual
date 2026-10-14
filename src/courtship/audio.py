"""Play audio into the fly's ears, and listen to what its wing does.

What this is NOT: the fly does not reproduce the input. Its wing motor
circuit drives a wing, not a speaker, and its output is a species-specific
motor pattern. Feeding in music and getting the music back out would mean
the connectome was acting as a transmission line, which it is not.

What it IS: both ends of this are real anatomy.

  INPUT. Johnston's organ (JO) neurons in the antenna are the fly's ears
  -- 559 of them in MaleCNS, 1,047 in FlyWire. The JO-A and JO-B subtypes
  are the vibration-sensitive ones, tuned to the near-field frequencies
  courtship song occupies (roughly 100-500 Hz, carrier near 250 Hz). The
  audio is band-passed to that range, and its envelope sets their firing
  rate -- which is how a real fly's antenna transduces sound.

  OUTPUT. Fly song is built from PULSES: each wing vibration produces a
  brief damped oscillation around 250 Hz. So a wing motor neuron spike
  maps onto one pulse, and the spike train becomes an audible waveform by
  construction rather than by analogy.

So the result is the fly's own response to what it heard -- it follows the
input's rhythm and intensity, not its pitch or melody. A female has no
wing-motor song circuit at all (and FlyWire, being brain-only, has no
motor neurons whatsoever), so her readout has to be her descending output
instead, which is a different quantity and is labelled as one.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert

# Johnston's organ is tuned to low-frequency near-field vibration; courtship
# song sits around 150-300 Hz. Anything outside this the fly cannot hear.
JO_BAND_HZ = (80.0, 1000.0)
# Fly pulse song: a brief damped oscillation per wing vibration.
PULSE_CARRIER_HZ = 250.0
PULSE_MS = 4.0


def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # winget installs it outside PATH for non-login shells
    for p in Path.home().glob(
        "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/**/bin/ffmpeg.exe"
    ):
        return str(p)
    return None


def load_audio(path: str | Path, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load any audio file as mono float32 in [-1, 1].

    WAV is read directly; anything else is converted through ffmpeg, which
    avoids adding a decoding dependency for a feature that is one step of
    one script.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() != ".wav":
        exe = _ffmpeg()
        if not exe:
            raise SystemExit(
                f"{path.suffix} needs ffmpeg to decode and it was not found. "
                "Convert the file to .wav yourself, or install ffmpeg."
            )
        tmp = Path(tempfile.gettempdir()) / f"courtship_in_{path.stem}.wav"
        subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", str(path),
             "-ac", "1", "-ar", str(target_sr), str(tmp)],
            check=True,
        )
        path = tmp

    sr, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak
    return data, int(sr)


def audio_to_firing_rate(
    samples: np.ndarray,
    sr: int,
    dt_ms: float,
    max_rate_hz: float = 180.0,
    band: tuple[float, float] = JO_BAND_HZ,
    waveform: bool = False,
) -> np.ndarray:
    """Convert a waveform into a per-timestep firing rate for JO neurons.

    Band-pass to what the fly can actually hear, take the analytic envelope,
    then resample to the simulation's timestep. The envelope is what the
    antenna transduces; the carrier itself is far above the rate a neuron
    can follow one-to-one.

    Two deliberate choices, both of which discard information:

    ABSOLUTE LEVEL IS DISCARDED. The envelope is normalised per clip, so a
    quiet recording drives the ears exactly as hard as a loud one. A real
    fly's response does depend on sound pressure, but nothing maps digital
    amplitude to SPL without knowing the playback level, so inventing one
    would be false precision. Only dynamics WITHIN a clip survive.

    DYNAMICS ARE COMPRESSED. The square root keeps loud passages from
    pinning every ear neuron at saturation, where differences in the input
    stop producing differences in the output.
    """
    nyq = sr / 2.0
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.99)
    if lo >= hi:
        raise ValueError(f"band {band} invalid for sample rate {sr}")
    b, a = butter(4, [lo, hi], btype="band")
    filtered = filtfilt(b, a, samples)

    if waveform:
        # Rate-code the SIGNAL, not its envelope: firing rate rides the
        # waveform itself, so the circuit receives frequency information
        # instead of only loudness. Measured 2026-10-14: the wing motor
        # spike density then peaks at the input frequency, 8/8 notes of a
        # tone sequence tracked to within 1%. Taking the envelope first
        # discards pitch before the connectome ever sees it.
        peak = np.abs(filtered).max()
        if peak > 0:
            filtered = filtered / peak
        n_steps = round(len(samples) / sr * 1000.0 / dt_ms)
        idx = np.linspace(0, len(filtered) - 1, n_steps)
        sig = np.interp(idx, np.arange(len(filtered)), filtered)
        # A rate cannot go negative, so centre it at half scale.
        return (max_rate_hz * (1.0 + sig) / 2.0).astype(np.float32)

    env = np.abs(hilbert(filtered))
    # Compress: loud passages should not pin every neuron at saturation.
    env = np.sqrt(np.maximum(env, 0.0))
    if env.max() > 0:
        env = env / env.max()

    n_steps = round(len(samples) / sr * 1000.0 / dt_ms)
    idx = np.linspace(0, len(env) - 1, n_steps)
    return (np.interp(idx, np.arange(len(env)), env) * max_rate_hz).astype(np.float32)


def spikes_to_song(
    counts_per_step: np.ndarray,
    dt_ms: float,
    sr: int = 22050,
    carrier_hz: float = PULSE_CARRIER_HZ,
    pulse_ms: float = PULSE_MS,
    n_neurons: int = 1,
) -> np.ndarray:
    """Turn a per-timestep population spike count into an audible waveform.

    Each timestep with activity emits one song pulse -- a damped sinusoid --
    with amplitude set by the fraction of the population that fired. This
    mirrors how the wing actually makes sound rather than sonifying the
    spike train by arbitrary convention.
    """
    pulse_n = max(4, int(sr * pulse_ms / 1000.0))
    t = np.arange(pulse_n) / sr
    pulse = np.sin(2 * np.pi * carrier_hz * t) * np.exp(-t / (pulse_ms / 3000.0))

    total = round(len(counts_per_step) * dt_ms / 1000.0 * sr) + pulse_n
    out = np.zeros(total, dtype=np.float32)
    denom = float(max(n_neurons, 1))
    for step, c in enumerate(counts_per_step):
        if c <= 0:
            continue
        at = int(step * dt_ms / 1000.0 * sr)
        amp = min(1.0, c / denom)
        out[at:at + pulse_n] += (pulse * amp).astype(np.float32)

    peak = np.abs(out).max()
    if peak > 0:
        out = out / peak * 0.92
    return out


def spikes_to_waveform(
    counts_per_step: np.ndarray,
    dt_ms: float,
    sr: int = 22050,
    band: tuple[float, float] = (60.0, 4000.0),
) -> np.ndarray:
    """Listen to the motor signal itself, instead of to the wing.

    `spikes_to_song` convolves every spike with a FIXED ~250 Hz pulse,
    which is physically what a wing does -- and which therefore overwrites
    whatever pitch the spike train carried with a constant carrier. That
    is right for "what would this fly sound like" and wrong for "what is
    the circuit actually transmitting".

    This renders the population spike density directly as the waveform, so
    the frequency content survives. Measured 2026-10-14: the wing motor
    population's spike density peaks at the INPUT frequency for 250, 400
    and 600 Hz tones (150 Hz comes out at its second harmonic, the usual
    rectification artefact). Pitch does reach the motor neurons.

    The real fly still cannot sing a melody: its wing carrier is set by
    mechanics, not by the nervous system. The distinction is between what
    the neurons encode and what the wing can physically radiate.
    """
    step_sr = 1000.0 / dt_ms
    n_out = round(len(counts_per_step) * dt_ms / 1000.0 * sr)
    x = np.asarray(counts_per_step, dtype=np.float64)
    x = x - x.mean()
    # Resample the spike density to audio rate.
    out = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)

    nyq = sr / 2.0
    hi = min(band[1], step_sr / 2.0 * 0.95, nyq * 0.99)
    lo = max(band[0], 1.0)
    if lo < hi:
        b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
        out = filtfilt(b, a, out)

    peak = np.abs(out).max()
    return (out / peak * 0.92).astype(np.float32) if peak > 0 else out.astype(np.float32)


def write_wav(path: str | Path, samples: np.ndarray, sr: int = 22050) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, (np.clip(samples, -1, 1) * 32767).astype(np.int16))
    return path
