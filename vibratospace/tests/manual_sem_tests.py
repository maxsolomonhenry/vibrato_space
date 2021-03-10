import os
import numpy as np

from vibratospace.src.python.defaults import DATA_PATH, SAMPLE_RATE, AUDIO_TEST_PATH
from vibratospace.src.python.oscillators import AdditiveOsc, Blit
from vibratospace.src.python.sem import EnvelopeInterpolator, FastConvolve
from vibratospace.src.python.spectral_util import get_spectral_envelope
from vibratospace.src.python.util import add_fade, load_data, midi_to_hz, normalize, time_plot, stft_plot


def sigmoid(x, alpha=1):
    temp = np.exp(-x * alpha)
    return (1 - temp)/(1 + temp)


if __name__ == '__main__':
    # Grab pre-calculated spectra.
    path = os.path.join(DATA_PATH, 'peak_valley_spectra.pickle')
    spectra = load_data(path)

    idx = 1

    peak = spectra[idx]['peak']
    mid = spectra[idx]['mid']
    valley = spectra[idx]['valley']

    # Filter order determines resolution of the spectral envelope.
    lpc_order = 120

    peak_envelope = get_spectral_envelope(peak, lpc_order=lpc_order)
    mid_envelope = get_spectral_envelope(mid, lpc_order=lpc_order)
    valley_envelope = get_spectral_envelope(valley, lpc_order=lpc_order)

    # Parameters.
    midi_note = 48.
    dur = 6

    # # High frequency weighting.
    # beta = 0.8
    # temp = np.linspace(1 - beta, 1 + beta, len(peak_envelope) // 2, endpoint=False)
    # spectral_weighting = np.concatenate([temp, temp[::-1]])

    # env1 = peak_envelope
    # env2 = mid_envelope
    # env3 = valley_envelope

    env1 = peak_envelope
    env2 = valley_envelope
    env3 = None

    import matplotlib.pyplot as plt
    plt.plot(np.abs(env1), label='1')
    plt.plot(np.abs(env2), label='2')
    # plt.plot(np.abs(env3), label='3')
    plt.legend()
    plt.show()

    hop_size = 512
    num_bins = len(env1)
    FRAME_RATE = int(np.round(SAMPLE_RATE / hop_size))

    out_filename = 'test.wav'

    # Synthesis...
    additive = AdditiveOsc(num_harmonics=30, randomize_phase=True)
    blit = Blit()

    osc = additive

    midi = np.tile(midi_note, int(np.round(dur * SAMPLE_RATE)))
    hz = midi_to_hz(midi)

    excitation_signal = osc(hz)
    # time_plot(excitation_signal)

    # excitation_signal += 0.01 * np.random.rand(len(excitation_signal))

    my_interp = EnvelopeInterpolator(env1, env2, env3)

    # 180 frames @ 512 samples per hop and 44100 Hz sample-rate => ~2 sec audio.
    mod_rate = 6
    test_mod = np.cos(2 * np.pi * mod_rate * np.linspace(0, dur, dur * FRAME_RATE))
    test_mod = sigmoid(test_mod, alpha=10)
    time_plot(test_mod)

    envelopes = my_interp(test_mod, emphasis=2)

    import matplotlib.pyplot as plt
    plt.imshow(envelopes, origin='lower', aspect='auto')
    plt.show()

    my_convolve = FastConvolve(
        frame_size=num_bins,
        hop_size=hop_size,
        debug=False
    )
    out_ = my_convolve(excitation_signal, envelopes)
    plt.plot(out_)
    plt.show()

    out_ -= np.mean(out_)
    out_ = normalize(out_)
    out_ = add_fade(out_, 0.25, SAMPLE_RATE)

    from scipy.io import wavfile

    path = os.path.join(AUDIO_TEST_PATH, out_filename)

    wavfile.write(path, SAMPLE_RATE, out_)
    stft_plot(out_, SAMPLE_RATE)
