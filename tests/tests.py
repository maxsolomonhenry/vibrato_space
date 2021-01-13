# Unit tests.

import glob
import os
from librosa.core import lpc
from scipy.io import wavfile
from scipy.signal import lfilter, freqz

from src.util import *
from src.defaults import SAMPLE_RATE
from src.synthesis import Blit, AdditiveOsc

PATH_TO_HERE = os.path.dirname(__file__)
path_pattern = os.path.join(PATH_TO_HERE, '../audio/raw/**/*.wav')
file_paths = glob.glob(path_pattern)


def my_read_wav(file_path):
    sample_rate, x = wavfile.read(file_path)
    x = force_mono(x)
    x = normalize(x)
    return sample_rate, x


def test_trim_silence(x, threshold=-5):
    y = trim_silence(x, threshold)

    time_plot(x, show=False)
    time_plot(y)


def test_lpc(x, order=12):
    x = trim_silence(x, -5)
    a = lpc(x, order=order)
    freqz(1, a, plot=plt.semilogx)
    plt.show()


if __name__ == '__main__':
    # for path in reversed(file_paths):
    #     sr, x = my_read_wav(path)

    #     test_trim_silence(x)
    #     test_lpc(x)

    # blit = Blit()
    # chirp = blit(np.linspace(440, 880, 2*SAMPLE_RATE, endpoint=False))
    # stft_plot(chirp)

    additive = AdditiveOsc(num_harmonics=25)
    chirp = additive(np.linspace(440, 880, 2*SAMPLE_RATE, endpoint=False))
    stft_plot(chirp)
