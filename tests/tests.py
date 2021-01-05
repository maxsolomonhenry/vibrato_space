# Unit tests.

import glob
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile

from src.util import *


PATH_TO_HERE = os.path.dirname(__file__)
path_pattern = os.path.join(PATH_TO_HERE, '../audio/raw/**/*.wav')
file_paths = glob.glob(path_pattern)

for path in reversed(file_paths):
    sample_rate, x = wavfile.read(path)
    x = force_mono(x)
    x = normalize(x)

    y = trim_silence(x, -5)

    time_plot(x, show=False)
    time_plot(y)