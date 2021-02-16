import os
import numpy as np

from vibratospace.src.python.defaults import DATA_PATH, SAMPLE_RATE, AUDIO_TEST_PATH
from vibratospace.src.python.oscillators import AdditiveOsc, Blit
from vibratospace.src.python.sem import EnvelopeInterpolator, FastConvolve
from vibratospace.src.python.spectral_util import get_spectral_envelope
from vibratospace.src.python.util import add_fade, load_data, midi_to_hz, normalize, time_plot

if __name__ == '__main__':
    # Grab pre-calculated spectra.
    path = os.path.join(DATA_PATH, 'peak_valley_spectra.pickle')
    spectra = load_data(path)

    peak = spectra[1]['peak']
    valley = spectra[1]['valley']

    peak_envelope = get_spectral_envelope(peak, lpc_order=120)
    valley_envelope = get_spectral_envelope(valley, lpc_order=120)

    # Parameters.
    midi_note = 50.
    dur = 2

    env1 = peak_envelope
    env2 = valley_envelope

    hop_size = 512
    num_bins = len(env1)

    out_filename = 'test.wav'

    # Synthesis...
    additive = AdditiveOsc(num_harmonics=25, randomize_phase=True)
    blit = Blit()

    osc = blit

    midi = np.tile(midi_note, dur * SAMPLE_RATE)
    hz = midi_to_hz(midi)

    excitation_signal = osc(hz)
    # time_plot(excitation_signal)

    # excitation_signal += 0.5 * np.random.rand(len(excitation_signal))

    my_interp = EnvelopeInterpolator(env1, env2)

    # 180 frames @ 512 samples per hop and 44100 Hz sample-rate => ~2 sec audio.
    test_mod = np.cos(2 * np.pi * 7 * np.linspace(0, 2, 173))
    test_mod = add_fade(test_mod, 0.6, SAMPLE_RATE//512)

    envelopes = my_interp(test_mod)

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
