"""
Utilities for pitch trajectory analysis and resynthesis.
"""

import librosa.display
import os
import pickle
import matlab
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
from vibratospace.src.python.defaults import EPS, SAMPLE_RATE, PITCH_RATE
from scipy.signal import hilbert, resample


def remove_dc(x: np.ndarray) -> np.ndarray:
    """
    Remove mean from signal.
    """
    return x - np.mean(x)


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


def fix_length(signal: np.ndarray, length: int) -> np.ndarray:
    """
    Pad or truncate array to specified length.
    """

    assert signal.ndim == 1
    if len(signal) < length:
        signal = np.pad(signal, [0, length - len(signal)])
    elif len(signal) > length:
        signal = signal[:length]
    assert signal.shape == (length,)
    return signal


def match_length(x1: np.ndarray, x2: np.ndarray) -> Tuple[np.array, np.array]:

    if len(x1) == len(x2):
        pass
    elif len(x1) < len(x2):
        x2 = fix_length(x2, len(x1))
    else:
        x1 = fix_length(x1, len(x2))
    return x1, x2


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


def to_sample_rate(input_):
    """
    Up-samples to project sample rate.
    """
    num_samples = int(
        round(len(input_) * SAMPLE_RATE / PITCH_RATE)
    )
    return resample(input_, num_samples)


def stft_plot(
    signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    title: str = "",
    show: bool = True
):
    X = librosa.stft(signal)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(5, 5))
    plt.title(title)
    librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="linear")
    if show:
        plt.show()


def fix_deviation(input_: np.ndarray, std: float):
    mean = np.mean(input_)

    input_ -= mean
    current_std = np.std(input_)
    input_ = input_ / current_std * std
    return input_ + mean


def widen(input_: np.ndarray, widen_amount: float):
    mean = np.mean(input_)
    input_ -= mean
    input_ *= widen_amount
    return input_ + mean


def load_data(path: str):
    assert os.path.isfile(path), 'Missing pickle. Run analysis.py'

    with open(path, 'rb') as handle:
        return pickle.load(handle)


def np2matlab(input_: np.ndarray):
    # Convert numpy array to Matlab double.
    return matlab.double(input_.tolist())[0]


def matlab2np(input_: matlab.double):
    # Convert Matlab double to numpy array.
    return np.array(input_._data)


def num2matlab(input_: Union[float, int]):
    return matlab.double([input_])
