# Misc. utilities as needed.
import numpy as np
import matplotlib.pyplot as plt
from defaults import EPS


def normalize(x: np.ndarray) -> np.ndarray:
    """
        Normalize array by max value.
    """
    return x/np.max(np.abs(x))


def time_plot(signal: np.ndarray, rate: int = 44100):
    t = np.linspace(0, len(signal)/rate, len(signal), endpoint=False)
    plt.plot(t, signal)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
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


def repitch(midi: np.ndarray, pitch: float=60) -> np.ndarray:
    """
        Recenters midi trajectory around a given pitch.
    """
    return midi - np.mean(midi) + pitch
