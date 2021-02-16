"""
Tools specifically dealing with spectra.
"""

import numpy as np
from librosa.core import lpc
from scipy.signal import lfilter


def get_spectral_envelope(spectrum: np.ndarray, lpc_order: int) -> np.ndarray:
    x = np.real(np.fft.ifft(spectrum))

    # Due to very specific complaint from `lpc`
    x = np.asfortranarray(x)

    a = lpc(x, order=lpc_order)
    impulse = np.zeros(len(x))
    impulse[0] = 1
    impulse_reponse = lfilter([1], a, impulse)
    return np.fft.fft(impulse_reponse)


if __name__ == '__main__':
    # Tests

    import os
    import matplotlib.pyplot as plt
    from vibratospace.src.python.util import load_data
    from vibratospace.src.python.defaults import DATA_PATH

    lpc_order = 80

    path = os.path.join(DATA_PATH, 'peak_valley_spectra.pickle')
    spectra = load_data(path)

    peak = spectra[0]['peak']
    valley = spectra[0]['valley']

    peak_env = get_spectral_envelope(peak, lpc_order)
    valley_env = get_spectral_envelope(valley, lpc_order)

    plt.plot(np.abs(peak)/np.max(np.abs(valley)), label='peak')
    plt.plot(np.abs(valley)/np.max(np.abs(valley)), label='valley')
    plt.plot(np.abs(peak_env)/np.max(np.abs(valley_env)), label='peak envelope')
    plt.plot(np.abs(valley_env)/np.max(np.abs(valley_env)), label='valley envelope')
    plt.legend()
    plt.show()
