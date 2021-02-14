"""
Script to check the pitch of synthesized audio.
"""

from vibratospace.src.python.defaults import PROCESSED_PATH, PITCH_RATE
from vibratospace.src.python.util import time_plot

import crepe
import pyworld
from glob import glob
import os
from scipy.io import wavfile

pattern = os.path.join(PROCESSED_PATH, '*.wav')
processed_files = glob(pattern)
assert processed_files, 'No files found matching pattern {}'.format(pattern)

for file in processed_files:
    sample_rate, x = wavfile.read(file)
    f0 = pyworld.harvest(x, sample_rate)[0]
    time_plot(f0, PITCH_RATE, show=True, title=os.path.basename(file))
