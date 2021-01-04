# Misc. utilities as needed.
import numpy as np
import matplotlib.pyplot as plt


def normalize(x: np.ndarray) -> np.ndarray:
    """
        Normalize array by max value.
    """
    return x/np.max(np.abs(x))


def wav_plot(signal: np.ndarray, rate: int = 44100):
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
