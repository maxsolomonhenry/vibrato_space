# Building signals from AR coefficients and pitch trajectories.
"""

    TODO:   - replace oscillator with BLIT (?) then filter.

"""

import numpy as np
import pickle
import os
from scipy.signal import lfilter, medfilt
from scipy.io import wavfile
from src.oscillators import Blit, AdditiveOsc
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


if __name__ == '__main__':
    VERBOSE = True

    # Median filter kernel is for separating jitter.
    median_filter_size = 5

    # Fix re-synthesis vibrato width as std in semitones.
    pitch_deviation = 0.15

    # Cosine ramp in/out for audio (in seconds).
    audio_fade = 0.25

    # Cosine ramp in/out for vibrato depth (in seconds).
    pitch_fade = 0.125

    # Note to recenter all stimuli (in midi: 48 == C3).
    repitch_note = 48

    assert os.path.isfile(PICKLE_PATH), 'Missing pickle. Run analysis.py'

    with open(PICKLE_PATH, 'rb') as handle:
        data = pickle.load(handle)

    # Ridiculous list comprehension to exclude voice-derived AR coefficients.
    average_coefficients = np.array([d['lpc'] for d in data if d['filename'].split('_')[0] != 'm1']).mean(axis=0)

    blit = Blit()
    additive = AdditiveOsc(num_harmonics=25)

    for datum in data:
        pitch = datum['frequency']
        pitch = hz_to_midi(pitch)
        pitch = repitch(pitch, repitch_note)

        # Remove and store jitter during pitch manipulation.
        jitter = pitch - medfilt(pitch, median_filter_size)
        pitch -= jitter

        pitch = add_fade(pitch, pitch_fade, rate=PITCH_RATE)
        pitch = fix_deviation(pitch, pitch_deviation)

        # Reintroduce jitter.
        pitch += jitter

        pitch = midi_to_hz(pitch)
        pitch = to_sample_rate(pitch)

        audio = additive(pitch)
        audio = lfilter([1], datum['lpc'], audio)

        audio = normalize(audio)
        audio = add_fade(audio, audio_fade, rate=SAMPLE_RATE)

        filename = "proc_" + datum['filename']
        write_path = os.path.join(PROCESSED_PATH, filename)

        if VERBOSE:
            print("Writing {}...".format(filename))
        wavfile.write(write_path, SAMPLE_RATE, audio)
