"""
    Excerpt extraction, pitch analysis, and LPC analysis.

    TODO: - argparse hell.
"""

import glob
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import IPython.display as ipd
import crepe

from util import normalize, force_mono, wav_plot
from defaults import PITCH_RATE

DEBUG = True

PATH_TO_HERE = os.path.dirname(__file__)
pitch_frame_dur = int(1/PITCH_RATE * 1000)
path_pattern = os.path.join(PATH_TO_HERE,'../audio/raw/**/*.wav')
file_paths = glob.glob(path_pattern)

if DEBUG:
    file_paths = [file_paths[0]]

for path in file_paths:
    sample_rate, x = wavfile.read(path)
    x = force_mono(x)
    x = normalize(x)

    time, frequency, confidence, activation = \
        crepe.predict(x, sample_rate, step_size=pitch_frame_dur)

    wav_plot(frequency, rate=PITCH_RATE)