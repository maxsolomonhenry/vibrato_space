# Misc. utilities as needed.
import numpy as np
import matplotlib.pyplot as plt
from src.defaults import EPS
from scipy.signal import hilbert


def normalize(x: np.ndarray) -> np.ndarray:
    """
        Normalize array by max value.
    """
    return x/np.max(np.abs(x))


def time_plot(signal: np.ndarray, rate: int = 44100, show: bool = True):
    t = np.linspace(0, len(signal)/rate, len(signal), endpoint=False)
    plt.plot(t, signal)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    if show:
        plt.show()


def force_mono(signal: np.ndarray) -> np.ndarray:
    """
        Forces stereo signal to mono by averaging channels.
    """
    assert len(signal.shape) <= 2, "Mono or stereo arrays only, please."
    if len(signal.shape) == 2:
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T
        signal = np.mean(signal, axis=0)
    return signal


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """
        Converts from Hz to linear pitch space, where midi:69 = A440.
    """
    return 12 * np.log2((hz + EPS)/440) + 69


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """
        Converts from linear pitch space to Hz, where A440 = midi:69.
    """
    return 440.0 * (2.0**((midi - 69.0) / 12.0))


def repitch(
    midi: np.ndarray,
    pitch: float = 60,
) -> np.ndarray:
    """
        Recenters midi trajectory around a given pitch.
    """
    return midi - np.mean(midi) + pitch


def add_fade(
    signal: np.ndarray,
    fade_length: float,
    rate: int
):
    """
        Adds raised cosine fade in/out to signal.
    """
    num_samples = int(fade_length * rate)

    # Build ramp.
    t = np.linspace(0, 0.5, num_samples, endpoint=False)
    ramp = 0.5 * np.cos(2 * np.pi * t) + 0.5

    mean = np.mean(signal)
    signal -= mean

    # Fade in/out.
    signal[:num_samples] *= ramp[::-1]
    signal[-num_samples:] *= ramp

    signal += mean

    return signal


def trim_excerpt(
    signal: np.ndarray,
    time_in: float = 1,
    duration: float = 1,
    rate: float = 44100
) -> np.ndarray:
    """
        Trim audio signal given start time and desired duration.
    """
    in_ = int(time_in * rate)
    out_ = in_ + int(duration * rate)
    return signal[in_:out_]


def trim_silence(
    signal: np.ndarray,
    threshold: float = -35,
    smoothing: int = 1024
) -> np.ndarray:
    """
        Trims beginning of audio signal until it passes a given threshold in dB.
    """
    amplitude_envelope = np.abs(hilbert(signal))
    smoothed = np.convolve(amplitude_envelope, np.ones(smoothing)/smoothing)
    log_envelope = np.log(smoothed + EPS)
    start_index = np.maximum(
        np.where(log_envelope >= threshold)[0][0] - smoothing//2,
        0
    )
    return signal[start_index:]