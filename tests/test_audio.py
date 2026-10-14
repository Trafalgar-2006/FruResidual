"""Tests for the audio -> ears -> wing -> audio path."""

import numpy as np
import pytest

from courtship.audio import (
    JO_BAND_HZ,
    audio_to_firing_rate,
    spikes_to_song,
    spikes_to_waveform,
    write_wav,
)


def tone(hz, seconds=0.5, sr=22050, amp=1.0):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32), sr


def test_firing_rate_has_one_value_per_timestep():
    x, sr = tone(250, seconds=1.0)
    rate = audio_to_firing_rate(x, sr, dt_ms=0.5)
    assert len(rate) == pytest.approx(2000, rel=0.01)  # 1000 ms / 0.5 ms
    assert rate.min() >= 0


def test_absolute_level_is_deliberately_discarded():
    # The envelope is normalised per clip, so a quiet recording drives the
    # ears as hard as a loud one. Nothing maps digital amplitude to sound
    # pressure without knowing playback level, so this is on purpose --
    # pinned here so it cannot change silently.
    loud, sr = tone(250, amp=1.0)
    quiet, _ = tone(250, amp=0.05)
    a = audio_to_firing_rate(loud, sr, 0.5).mean()
    b = audio_to_firing_rate(quiet, sr, 0.5).mean()
    assert a == pytest.approx(b, rel=0.01), (a, b)


def test_inaudible_frequency_is_filtered_out():
    # The fly cannot hear 8 kHz; Johnston's organ is tuned far below it.
    # A tone outside the band must drive the ears far less than one inside.
    inside, sr = tone(250)
    outside, _ = tone(8000)
    r_in = audio_to_firing_rate(inside, sr, 0.5).mean()
    r_out = audio_to_firing_rate(outside, sr, 0.5).mean()
    assert r_out < r_in * 0.2, (r_in, r_out)
    assert JO_BAND_HZ[1] < 8000


def test_rate_tracks_the_envelope_not_the_carrier():
    # Same carrier, amplitude ramped: the drive must rise with the ramp.
    sr = 22050
    t = np.arange(sr) / sr
    x = (np.sin(2 * np.pi * 250 * t) * np.linspace(0, 1, sr)).astype(np.float32)
    rate = audio_to_firing_rate(x, sr, 0.5)
    first, last = rate[: len(rate) // 4].mean(), rate[-len(rate) // 4:].mean()
    # Only 2x, not the ~7x of the raw ramp: the square-root compression in
    # audio_to_firing_rate deliberately flattens dynamics so loud passages
    # don't saturate every ear neuron. Direction is what matters here.
    assert last > first * 2, (first, last)


def test_silence_in_means_silence_out():
    counts = np.zeros(400)
    wav = spikes_to_song(counts, dt_ms=0.5, n_neurons=10)
    assert np.abs(wav).max() == 0


def test_more_spikes_make_a_louder_signal():
    sparse = np.zeros(400); sparse[::50] = 1
    dense = np.zeros(400); dense[::5] = 1
    a = spikes_to_song(sparse, 0.5, n_neurons=1)
    b = spikes_to_song(dense, 0.5, n_neurons=1)
    assert np.sqrt((b ** 2).mean()) > np.sqrt((a ** 2).mean())


def test_song_length_matches_the_spike_train_duration():
    counts = np.zeros(2000); counts[::100] = 1   # 2000 steps x 0.5 ms = 1 s
    sr = 22050
    wav = spikes_to_song(counts, 0.5, sr=sr, n_neurons=1)
    assert len(wav) / sr == pytest.approx(1.0, abs=0.02)


def test_pulses_land_where_the_spikes_are():
    counts = np.zeros(2000)
    counts[1000] = 1                      # a single spike at 500 ms
    sr = 22050
    wav = spikes_to_song(counts, 0.5, sr=sr, n_neurons=1)
    loudest = int(np.argmax(np.abs(wav)))
    assert loudest / sr == pytest.approx(0.5, abs=0.01)


def test_wav_roundtrips(tmp_path):
    from scipy.io import wavfile
    wav = spikes_to_song(np.array([0, 1, 0, 1] * 100), 0.5, n_neurons=1)
    p = write_wav(tmp_path / "x.wav", wav)
    sr, back = wavfile.read(p)
    assert sr == 22050
    assert len(back) == len(wav)


def test_waveform_mode_carries_frequency_envelope_mode_does_not():
    # The whole point of waveform mode: the drive must oscillate at the
    # input's frequency. Envelope mode gives a near-constant rate for a
    # steady tone, which is why it cannot carry pitch.
    x, sr = tone(250, seconds=0.3)
    wave = audio_to_firing_rate(x, sr, 0.1, waveform=True)
    env = audio_to_firing_rate(x, sr, 0.1, waveform=False)

    def peak_hz(sig, step_sr):
        sig = sig - sig.mean()
        spec = np.abs(np.fft.rfft(sig))
        fr = np.fft.rfftfreq(len(sig), 1 / step_sr)
        band = fr > 20
        return fr[band][np.argmax(spec[band])]

    assert peak_hz(wave, 10000) == pytest.approx(250, rel=0.1)
    assert wave.std() > env.std() * 5


def test_waveform_drive_stays_non_negative():
    x, sr = tone(250)
    assert audio_to_firing_rate(x, sr, 0.1, waveform=True).min() >= 0


def test_density_rendering_preserves_input_frequency():
    # A spike density oscillating at 300 Hz must render to audio whose
    # dominant frequency is 300 Hz -- unlike spikes_to_song, which imposes
    # its own fixed carrier on everything.
    dt, n = 0.1, 6000              # 0.6 s at 10 kHz
    t = np.arange(n) * dt / 1000.0
    counts = np.maximum(0, np.sin(2 * np.pi * 300 * t)) * 5
    sr = 22050
    wav = spikes_to_waveform(counts, dt, sr=sr)
    spec = np.abs(np.fft.rfft(wav))
    fr = np.fft.rfftfreq(len(wav), 1 / sr)
    band = (fr > 80) & (fr < 2000)
    assert fr[band][np.argmax(spec[band])] == pytest.approx(300, rel=0.08)


def test_pulse_rendering_also_keeps_the_rate_but_adds_carrier_colour():
    # Both renderings keep the repetition frequency -- the pulse RATE is
    # itself the signal, which is also how real fly song encodes structure.
    # The difference is timbre: the pulse version adds energy at its own
    # 250 Hz carrier that was never in the input. Pitch is lost on the
    # INPUT side by taking an envelope, not here.
    dt, n = 0.1, 6000
    t = np.arange(n) * dt / 1000.0
    counts = np.maximum(0, np.sin(2 * np.pi * 300 * t)) * 5
    sr = 22050

    def spectrum(wav):
        spec = np.abs(np.fft.rfft(wav))
        fr = np.fft.rfftfreq(len(wav), 1 / sr)
        band = (fr > 80) & (fr < 2000)
        return fr[band], spec[band]

    fr_p, sp_p = spectrum(spikes_to_song(counts, dt, sr=sr, n_neurons=5))
    fr_d, sp_d = spectrum(spikes_to_waveform(counts, dt, sr=sr))
    assert fr_p[np.argmax(sp_p)] == pytest.approx(300, rel=0.08)
    assert fr_d[np.argmax(sp_d)] == pytest.approx(300, rel=0.08)

    def carrier_share(fr, sp):
        return sp[(fr > 230) & (fr < 270)].sum() / sp.sum()

    assert carrier_share(fr_p, sp_p) > carrier_share(fr_d, sp_d)
