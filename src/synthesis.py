# Building signals from AR coefficients and pitch trajectories.
"""

    TODO:   - replace oscillator with BLIT (?) then filter.

"""

import numpy as np
import pickle
import os
from scipy.signal import lfilter
from scipy.io import wavfile
from src.util import (
    midi_to_hz,
    to_sample_rate,
    normalize,
    add_fade,
    fix_deviation,
    hz_to_midi,
    repitch
)
from src.defaults import (
    PICKLE_PATH,
    PROCESSED_PATH,
    SAMPLE_RATE,
    EPS,
    PITCH_RATE
)


class Blit:
    """
    Band-limited impulse train.

    This code basically a carbon copy of STK's BLIT class. Thanks Gary.
    """

    def __init__(self):
        self.phase = 0
        self.output = []

    def __call__(self, hz: np.ndarray) -> np.ndarray:
        self.reset()

        for frequency in hz:
            p_, rate_ = self.set_frequency(frequency)
            m_ = self.update_harmonics(p_)

            denominator = np.sin(np.pi * self.phase)

            if denominator <= EPS:
                temp = 1
            else:
                temp = np.sin(m_ * np.pi * self.phase)
                temp /= m_ * denominator

            self.phase += rate_

            if self.phase >= 1:
                self.phase -= 1

            self.output.append(temp)
        return np.array(self.output) - np.mean(self.output)

    def reset(self):
        self.phase = 0
        self.output = []

    @staticmethod
    def set_frequency(frequency):
        p_ = SAMPLE_RATE / frequency
        rate_ = 1 / p_
        return p_, rate_

    @staticmethod
    def update_harmonics(p_):
        max_harmonics = np.floor(0.5 * p_)
        return 2 * max_harmonics + 1


if __name__ == '__main__':
    VERBOSE = True

    pitch_deviation = 0.15
    audio_fade = 0.25
    pitch_fade = 0.125
    repitch_note = 52

    assert os.path.isfile(PICKLE_PATH), 'Missing pickle. Run analysis.py'

    with open(PICKLE_PATH, 'rb') as handle:
        data = pickle.load(handle)

    # Ridiculous list comprehension to exclude voice-derived AR coefficients.
    average_coefficients = np.array([d['lpc'] for d in data if d['filename'].split('_')[0] != 'm1']).mean(axis=0)

    blit = Blit()

    for datum in data:
        pitch = datum['frequency']
        pitch = hz_to_midi(pitch)
        pitch = repitch(pitch, repitch_note)
        pitch = add_fade(pitch, pitch_fade, rate=PITCH_RATE)

        pitch = fix_deviation(pitch, pitch_deviation)
        pitch = midi_to_hz(pitch)
        pitch = to_sample_rate(pitch)

        audio = blit(pitch)
        audio = lfilter([1], average_coefficients, audio)
        # audio = lfilter([1], data[6]['lpc'], audio)

        audio = normalize(audio)
        audio = add_fade(audio, audio_fade, rate=SAMPLE_RATE)

        filename = "proc_" + datum['filename']
        write_path = os.path.join(PROCESSED_PATH, filename)

        if VERBOSE:
            print("Writing {}...".format(filename))
        wavfile.write(write_path, SAMPLE_RATE, audio)
