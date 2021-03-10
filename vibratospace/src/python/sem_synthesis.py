import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import os

from vibratospace.src.python.defaults import DATA_PATH, PITCH_RATE, SAMPLE_RATE
from vibratospace.src.python.oscillators import AdditiveOsc, Blit
from vibratospace.src.python.sem_engine import FastConvolve
from vibratospace.src.python.util import load_data, midi_to_hz, stft_plot


def half_to_full(spectrum):
    """
    Quick method to make symmetrical spectra from real-only spectra.
    """
    num_frames = spectrum.shape[0]

    half_frame_size = spectrum.shape[1]
    full_frame_size = (half_frame_size - 1) * 2

    temp = np.zeros([num_frames, full_frame_size])
    temp[:, :half_frame_size] = spectrum
    temp[:, half_frame_size:] = np.flip(spectrum[:, 1:-1], axis=1)

    return temp


if __name__ == '__main__':
    data = load_data(os.path.join(DATA_PATH, 'data.pickle'))

    # TODO: iterate through data.
    datum = data[0]

    spectral_envelopes = datum['world']['sp']
    spectral_envelopes = np.sqrt(spectral_envelopes)
    spectral_envelopes = half_to_full(spectral_envelopes)

    my_convolve = FastConvolve(
        frame_size=spectral_envelopes.shape[1],
        hop_size=spectral_envelopes.shape[1]//8
    )

    blit = Blit()
    additive = AdditiveOsc(num_harmonics=20)
    f0 = midi_to_hz(50)
    carrier = additive(np.tile(f0, int(SAMPLE_RATE * 1.75)))

    x = my_convolve(carrier, spectral_envelopes)

    display(ipd.Audio(x, rate=SAMPLE_RATE))
    display(ipd.Audio(datum['audio'], rate=SAMPLE_RATE))

    stft_plot(x)
    stft_plot(datum['audio'])


