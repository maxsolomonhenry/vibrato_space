"""
    Excerpt extraction, pitch analysis, and LPC analysis.

    TODO: - argparse hell.
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import crepe

from util import normalize, force_mono, time_plot, hz_to_midi, repitch
from defaults import PITCH_RATE

DEBUG = False

PATH_TO_HERE = os.path.dirname(__file__)
pitch_frame_dur = int(1/PITCH_RATE * 1000)
path_pattern = os.path.join(PATH_TO_HERE, '../audio/raw/**/*.wav')
file_paths = glob.glob(path_pattern)

if DEBUG:
    file_paths = [file_paths[-1]]

for path in file_paths:
    sample_rate, x = wavfile.read(path)
    x = force_mono(x)
    x = normalize(x)

    frequency = crepe.predict(x, sample_rate, step_size=pitch_frame_dur)[1]

    midi = hz_to_midi(frequency)
    midi = midi[125:325]
    midi = repitch(midi, 60)
    plt.plot(midi)