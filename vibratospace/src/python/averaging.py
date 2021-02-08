"""
Generate an average spectrum given a list of AR coefficients.

Because interpolation has to happen in the spectral domain, this takes a bit of
finagling and its not as simple as it would seem.
"""

import numpy as np
from librosa.core import lpc
from scipy.signal import lfilter, freqz


def impulse_of_length(length: int) -> np.ndarray:
    temp = np.zeros(length)
    temp[0] = 1
    return temp


def ar_to_spectrum(a: np.ndarray, length: int = 1024):
    impulse_response = lfilter([1], a, impulse_of_length(length))
    X = np.fft.rfft(impulse_response)
    return np.abs(X)


def batch_ar_to_spectra(list_of_ar_coefficients: list, N: int = 1024):
    spectra = []

    for ar_coefficients in list_of_ar_coefficients:
        spectra.append(ar_to_spectrum(ar_coefficients, N))

    return spectra


def get_average_spectrum(spectra: list) -> np.ndarray:
    temp = np.vstack(spectra)
    return temp.mean(axis=0)


def randomize_phase(positive_spectrum: np.ndarray) -> np.ndarray:
    hN = len(positive_spectrum)
    mX = np.abs(positive_spectrum)
    random_phase = np.random.rand(hN) * 2 * np.pi
    return mX * np.exp(1j * random_phase)


def get_average_ar_coefficients(batch_coefficients: list) -> np.ndarray:
    filter_order = len(batch_coefficients[0]) - 1
    spectra = batch_ar_to_spectra(batch_coefficients)
    average_spectrum = get_average_spectrum(spectra)

    # IFFT of spectrum gives the auto-correlation function.
    autocorrelation = np.fft.irfft(average_spectrum)

    return lpc(autocorrelation, filter_order)


if __name__ == "__main__":
    # Tests.

    import matplotlib.pyplot as plt
    import os
    from vibratospace.src.python.util import load_data
    from vibratospace.src.python.defaults import DATA_PATH

    pickle_path = os.path.join(DATA_PATH, 'data.pickle')
    pickle_path = os.path.abspath(pickle_path)
    data = load_data(pickle_path)

    batch_coefficients = [d['lpc'] for d in data]

    average_coefficients = get_average_ar_coefficients(batch_coefficients)

    w, h = freqz([1], average_coefficients)
    plt.plot(w, np.abs(h))
    plt.show()
