"""
    Excerpt extraction, pitch analysis, and LPC analysis.

    TODO:   - argparse hell.
            - LPC analysis (add to dict?)
"""

import glob
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import crepe

from src.defaults import PITCH_RATE
from src.util import (
    normalize,
    force_mono,
    hz_to_midi,
    repitch,
    add_fade,
    trim_excerpt,
    trim_silence
)

DEBUG = True

# Parameters for clipping pitch trajectories.
excerpt_in = 0.5
excerpt_duration = 1.75
pitch_fade = 0.125

# Calculation for CREPE.
pitch_frame_dur = int(1/PITCH_RATE * 1000)

# Find audio files.
PATH_TO_HERE = os.path.dirname(__file__)
path_pattern = os.path.join(PATH_TO_HERE, '../audio/raw/**/*.wav')
file_paths = glob.glob(path_pattern)

trajectories = []

for path in file_paths:
    sample_rate, x = wavfile.read(path)

    x = force_mono(x)
    x = normalize(x)
    x = trim_silence(x, -5)
    x = trim_excerpt(x, excerpt_in, excerpt_duration, rate=sample_rate)

    frequency = crepe.predict(x, sample_rate, step_size=pitch_frame_dur)[1]

    midi = hz_to_midi(frequency)
    midi = repitch(midi, 60)
    midi = add_fade(midi, pitch_fade, rate=PITCH_RATE)

    trajectories.append({'filename': os.path.basename(path), 'midi': midi})

    if DEBUG:
        plt.plot(midi)

if DEBUG:
    plt.show()
