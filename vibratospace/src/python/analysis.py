"""
Excerpt extraction, pitch analysis, WORLD analysis, and LPC analysis.

The script will scan all audio in the `RAW_AUDIO_PATH` directory (and
sub-directories). Saves data as a .pickle file in `PICKLE_PATH`. All paths
specified in `defaults.py`.

Uses CREPE for pitch track.

Also uses the WORLD algorithm to generate f0 and spectral envelope.

Kim, Jong Wook, Justin Salamon, Peter Li, and Juan Pablo Bello. "Crepe: A
    convolutional representation for pitch estimation." In 2018 IEEE
    International Conference on Acoustics, Speech and Signal Processing
    (ICASSP), pp. 161-165, 2018.

M. Morise, F. Yokomori, and K. Ozawa, ``WORLD: a vocoder-based high-quality
    speech synthesis system for real-time applications,'' IEICE transactions on
    information and systems, vol. E99-D, no. 7, pp. 1877-1884, 2016.

    TODO:   - add log.txt (?)
"""

import argparse
import glob
import os
import crepe
import pickle
import warnings
import pyworld as pw
from scipy.io import wavfile
from librosa.core import lpc
from tqdm import tqdm

from vibratospace.src.python.defaults import (
    DATA_PATH,
    PITCH_RATE,
    RAW_AUDIO_PATH
)
from vibratospace.src.python.util import (
    normalize,
    force_mono,
    trim_excerpt,
    trim_silence
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch generate pitch trajectories and spectral envelopes.'
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
    parser.add_argument(
        '-v',
        '--verbose',
        type=bool,
        default=True,
        help='Let us talk through it.'
    )
    args = parser.parse_args()

    # Calculation for CREPE.
    pitch_frame_dur = 1/PITCH_RATE * 1000

    # Find audio files.
    PATH_TO_HERE = os.path.dirname(__file__)
    path_pattern = os.path.join(PATH_TO_HERE, RAW_AUDIO_PATH)
    file_paths = glob.glob(path_pattern)
    assert file_paths, "Could not find any files matching:\t{}".format(path_pattern)


    data = []

    for path in tqdm(file_paths):
        if args.verbose:
            tqdm.write('Reading {}...'.format(os.path.basename(path)))

        # Suppress wavfile complaints.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sample_rate, x = wavfile.read(path)

        x = force_mono(x)
        x = normalize(x)
        x = trim_silence(x, threshold=-5)

        if args.excerpt_duration:
            x = trim_excerpt(
                x,
                args.excerpt_in,
                args.excerpt_duration,
                rate=sample_rate
            )

        if args.verbose:
            tqdm.write('Tracking pitch with CREPE...')
        crepe_f0 = crepe.predict(
            x,
            sample_rate,
            viterbi=True,
            step_size=pitch_frame_dur,
            verbose=False
        )[1]

        if args.verbose:
            tqdm.write('Estimating AR coefficients...')
        lpc_coefficients = lpc(x, order=args.lpc_order)

        if args.verbose:
            tqdm.write('WORLD analysis...')
        f0, sp, ap = pw.wav2world(x, sample_rate)

        world_data = {
            'f0': f0,
            'sp': sp,
            'ap': ap,
        }

        data.append(
            {
                'filename': os.path.basename(path),
                'sample_rate': sample_rate,
                'audio': x,
                'crepe_f0': crepe_f0,
                'lpc': lpc_coefficients,
                'world': world_data
            }
        )

    pickle_path = os.path.join(DATA_PATH, 'data.pickle')
    pickle_path = os.path.abspath(pickle_path)

    if args.make_pickle:
        tqdm.write('Saving pickle at {}'.format(pickle_path))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
