import numpy as np
from scipy.interpolate import interp1d
from vibratospace.src.defaults import SAMPLE_RATE, PITCH_RATE


def upsample_shell(hz: np.ndarray, kind: str) -> np.ndarray:
    """
    Convenience function for cubic interpolation.
    """

    upsampled_indices = np.arange(len(hz)) * SAMPLE_RATE / PITCH_RATE
    f = interp1d(upsampled_indices, hz, kind=kind)

    num_samples = int(
        round((len(hz) - 1) * SAMPLE_RATE / PITCH_RATE)
    )

    return f(np.arange(num_samples))


def linear(hz: np.ndarray) -> np.ndarray:
    return upsample_shell(hz, kind='linear')


def quadratic(hz: np.ndarray) -> np.ndarray:
    return upsample_shell(hz, kind='quadratic')


def cubic(hz: np.ndarray) -> np.ndarray:
    return upsample_shell(hz, kind='cubic')


def next_(hz: np.ndarray) -> np.ndarray:
    return upsample_shell(hz, kind='next')


def prev_(hz: np.ndarray) -> np.ndarray:
    return upsample_shell(hz, kind='previous')


def nearest(hz: np.ndarray) -> np.ndarray:
    return upsample_shell(hz, kind='nearest')


if __name__ == '__main__':
    from vibratospace.src.util import time_plot, to_sample_rate

    # Simulate jitter about 440 Hz.
    test = (np.random.rand(100) * 2 - 1) + 440

    # Compare cubic upsample to standard.
    cubic_test = cubic_upsample(test)
    normal_test = to_sample_rate(test)

    time_plot(cubic_test, show=False)
    time_plot(normal_test)