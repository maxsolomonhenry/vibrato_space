"""
    Excerpt extraction, pitch analysis, and LPC analysis.

    TODO:   √ argparse hell.
            √ LPC analysis (add to dict?)
            - add log.txt
            - suppress warnings
"""

import argparse
import glob
import os
import crepe
import pickle
import matplotlib.pyplot as plt
from scipy.io import wavfile
from librosa.core import lpc
from tqdm import tqdm

from vibratospace.src.defaults import PITCH_RATE, PICKLE_PATH, RAW_AUDIO_PATH
from vibratospace.src.util import (
    normalize,
    force_mono,
    trim_excerpt,
    trim_silence
)

parser = argparse.ArgumentParser(
    description='Batch generate pitch trajectories and spectral envelopes.'
)
parser.add_argument(
    '-D',
    '--DEBUG',
    type=bool,
    default=False,
    help='Show plots.'
)
parser.add_argument(
    '-in',
    '--excerpt_in',
    type=float,
    default=0.75,
    help='Start time of excerpt (after trimmed silence) in seconds.'
)
parser.add_argument(
    '-d',
    '--excerpt_duration',
    type=float,
    default=1.75,
    help='Duration of pitch trajectory in seconds.'
)
parser.add_argument(
    '-a',
    '--lpc_order',
    type=int,
    default=50,
    help='The number of AR coefficients to predict.'
)
parser.add_argument(
    '-s',
    '--make_pickle',
    type=bool,
    default=True,
    help='Save a pickle in default path.'
)
args = parser.parse_args()

# Calculation for CREPE.
pitch_frame_dur = 1/PITCH_RATE * 1000

# Find audio files.
PATH_TO_HERE = os.path.dirname(__file__)
path_pattern = os.path.join(PATH_TO_HERE, RAW_AUDIO_PATH)
file_paths = glob.glob(path_pattern)

data = []

for path in tqdm(file_paths):
    sample_rate, x = wavfile.read(path)

    x = force_mono(x)
    x = normalize(x)
    x = trim_silence(x, threshold=-5)
    x = trim_excerpt(x, args.excerpt_in, args.excerpt_duration, rate=sample_rate)

    frequency = crepe.predict(
        x,
        sample_rate,
        viterbi=True,
        step_size=pitch_frame_dur,
        verbose=args.DEBUG
    )[1]

    lpc_coefficients = lpc(x, order=args.lpc_order)

    data.append(
        {
            'filename': os.path.basename(path),
            'audio': x,
            'frequency': frequency,
            'lpc': lpc_coefficients
        }
    )

    if args.DEBUG:
        plt.plot(frequency)

if args.DEBUG:
    plt.show()

if args.make_pickle:
    print('Saving pickle at {}'.format(PICKLE_PATH))
    with open(PICKLE_PATH, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
