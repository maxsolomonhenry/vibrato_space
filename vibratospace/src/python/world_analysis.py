"""
Uses the WORLD algorithm to generate f0 and spectral envelope.

The script will scan all audio in the `RAW_AUDIO_PATH` directory (and
sub-directories). Saves data as a .pickle file in `PICKLE_PATH`.

All paths specified in `defaults.py`.

M. Morise, F. Yokomori, and K. Ozawa, ``WORLD: a vocoder-based high-quality
    speech synthesis system for real-time applications,'' IEICE transactions on
    information and systems, vol. E99-D, no. 7, pp. 1877-1884, 2016.
"""

# TODO: factor analysis from both scripts?

import glob
import os
import pickle
import pyworld as pw

from scipy.io import wavfile
from tqdm import tqdm
from vibratospace.src.python.defaults import (
    AUDIO_TEST_PATH,
    DATA_PATH,
    RAW_AUDIO_PATH,
)
from vibratospace.src.python.util import (
    force_mono,
    match_length,
    normalize,
    time_plot,
    trim_excerpt,
    trim_silence
)

# Analysis parameters.
excerpt_in = .75
excerpt_duration = 1.75


# Find audio files.
PATH_TO_HERE = os.path.dirname(__file__)
path_pattern = os.path.join(PATH_TO_HERE, RAW_AUDIO_PATH)
path_pattern = os.path.abspath(path_pattern)
file_paths = glob.glob(path_pattern)
assert file_paths, "Could not find any files matching:\t{}".format(path_pattern)

data = []

for path in tqdm(file_paths):
    filename = os.path.basename(path)
    sample_rate, x = wavfile.read(path)

    x = force_mono(x)
    x = normalize(x)
    x = trim_silence(x, threshold=-5)
    x = trim_excerpt(x, excerpt_in, excerpt_duration, rate=sample_rate)

    # Analyze with WORLD.
    f0, sp, ap = pw.wav2world(x, sample_rate)

    data.append(
        {
            'filename': os.path.basename(path),
            'audio': x,
            'frequency': f0,
            'sp': sp,
            'ap': ap,
            'sample_rate': sample_rate
        }
    )


pickle_path = os.path.join(DATA_PATH, 'world_data.pickle')
pickle_path = os.path.abspath(pickle_path)

print('Saving pickle at {}'.format(pickle_path))
with open(pickle_path, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
