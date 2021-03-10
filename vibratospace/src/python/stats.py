# +
"""
Quick script to get the average vibrato extent of the cello tracks.
"""

import numpy as np
import os

from vibratospace.src.python.defaults import DATA_PATH, PITCH_RATE
from vibratospace.src.python.util import hz_to_midi, load_data, time_plot


path = os.path.join(DATA_PATH, 'data.pickle')
data = load_data(path)

# Buffer to crop in, remove artifacts of pitch tracking.
buf = 10

widths = []

for datum in data:
    if datum['filename'].split('_')[0] == 'VC':
        pitch = datum['world']['f0']
        pitch = hz_to_midi(pitch)
        pitch = pitch[buf:-buf]
        
        tmp_min = min(pitch)
        tmp_max = max(pitch)
        
        widths.append(tmp_max - tmp_min)
        
        time_plot(pitch, rate=PITCH_RATE)
        
print("Mean vibrato width = {}".format(np.mean(widths)/2))

# +
# Testing some equations...

mod_5hz = np.cos(2*np.pi*5*np.linspace(0, 1, 44100))

test = 2 ** (24 * mod_5hz/12) * 1000

time_plot(test)
# -


