"""
Synthesizes audio with manipulated pitch trajectories.

Pitch trajectory and spectral envelope information are stored in a `.pickle`
file in the `DATA_PATH` directory. To generate this file, run `analysis.py`.

World synthesis uses the WORLD vocoder:

M. Morise, F. Yokomori, and K. Ozawa, ``WORLD: a vocoder-based high-quality
    speech synthesis system for real-time applications,'' IEICE transactions on
    information and systems, vol. E99-D, no. 7, pp. 1877-1884, 2016.
"""

import argparse
import os
import matlab.engine
import pyworld as pw
from scipy.signal import lfilter, medfilt
from scipy.io import wavfile
from vibratospace.src.python.averaging import get_average_ar_coefficients
from vibratospace.src.python.oscillators import Blit, AdditiveOsc
from vibratospace.src.python.defaults import (
    DATA_PATH,
    MATLAB_SCRIPT_PATH,
    PITCH_RATE,
    PROCESSED_PATH,
    SAMPLE_RATE,
    SM_PATH
)
from vibratospace.src.python.util import (
    add_fade,
    fix_deviation,
    hz_to_midi,
    load_data,
    matlab2np,
    midi_to_hz,
    num2matlab,
    np2matlab,
    normalize,
    remove_dc,
    repitch,
    to_sample_rate,
    widen
)

# TODO: - filename flags, or log, for synthesis.
#       - argparser


def sine_model_resynth(pitch, sample_rate, num_partials, SM_PATH):
    """
    Wrapper for Matlab call to SM model.
    """
    pitch = np2matlab(pitch)
    sample_rate = num2matlab(sample_rate)
    num_partials = num2matlab(num_partials)

    pitch = eng.sineModelResynthesis(
        pitch,
        sample_rate,
        num_partials,
        SM_PATH
    )

    return matlab2np(pitch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch re-synthesis of pitch trajectories.'
    )
    parser.add_argument(
        '-s',
        '--synth_engine',
        type=str,
        default='blit',
        help='Specify synthesis engine: [\'blit\', \'additive\', \'world\'].'
    )
    parser.add_argument(
        '-m',
        '--median_filter_size',
        type=int,
        default=None,
        help='Median filter kernel size.'
    )
    parser.add_argument(
        '-d',
        '--pitch_std',
        type=int,
        default=None,
        help='Fix re-synthesis vibrato width as std in semitones.'
    )
    parser.add_argument(
        '-w',
        '--pitch_widen',
        type=float,
        default=None,
        help='Expand the pitch trajectory by a multiplicative factor.'
    )
    parser.add_argument(
        '-f',
        '--audio_fade',
        type=float,
        default=0.125,
        help='Cosine ramp in/out for audio (in seconds).'
    )
    parser.add_argument(
        '-p',
        '--pitch_fade',
        type=float,
        default=None,
        help='Cosine ramp in/out for vibrato depth (in seconds).'
    )
    parser.add_argument(
        '-r',
        '--repitch_note',
        type=float,
        default=None,
        help='Note to recenter all stimuli (in midi: 48 == C3).'
    )
    parser.add_argument(
        '-n',
        '--num_partials',
        type=int,
        default=None,
        help='Number of partials to reconstruct pitch from sinusoidal model'
    )
    args = parser.parse_args()

    synth_engine = args.synth_engine
    assert synth_engine in ['blit', 'additive', 'world'], "Unrecognized engine."

    median_filter_size = args.median_filter_size  # 3
    pitch_std = args.pitch_std
    pitch_widen = args.pitch_widen
    audio_fade = args.audio_fade
    pitch_fade = args.pitch_fade
    repitch_note = args.repitch_note
    num_partials = args.num_partials

    # Load data (generated by analysis.py).
    pickle_path = os.path.join(DATA_PATH, 'data.pickle')
    pickle_path = os.path.abspath(pickle_path)
    data = load_data(pickle_path)

    # Start Matlab engine.
    if num_partials:
        print('Starting Matlab engine...')
        eng = matlab.engine.start_matlab()
        eng.addpath(os.path.realpath(MATLAB_SCRIPT_PATH))

    # Calculate average spectrum and generate ar coefficients.
    if synth_engine is not 'world':
        batch_coefficients = [d['lpc'] for d in data]
        average_coefficients = get_average_ar_coefficients(batch_coefficients)

        # # Add negative control.
        # blank_frequency = np.ones(len(data[0]['frequency']))
        #
        # data.append(
        #     {
        #         'filename': 'no_vibrato.wav',
        #         'crepe_f0': blank_frequency
        #     }
        # )

    # Init oscillators.
    blit = Blit()
    additive = AdditiveOsc(num_harmonics=25)

    for datum in data:
        if synth_engine == 'world':
            pitch = datum['world']['f0']
        else:
            pitch = datum['crepe_f0']

        pitch = hz_to_midi(pitch)

        if repitch_note:
            pitch = repitch(pitch, repitch_note)

        # Remove and store jitter during pitch manipulation.
        if median_filter_size:
            jitter = pitch - medfilt(pitch, median_filter_size)
            pitch -= jitter

        if pitch_fade:
            pitch = add_fade(pitch, pitch_fade, rate=PITCH_RATE)

        if pitch_std:
            pitch = fix_deviation(pitch, pitch_std)

        if pitch_widen:
            pitch = widen(pitch, pitch_widen)

        # Reintroduce jitter.
        if median_filter_size:
            pitch += jitter

        pitch = midi_to_hz(pitch)

        if num_partials:
            pitch = sine_model_resynth(pitch, PITCH_RATE, num_partials, SM_PATH)

        if synth_engine == 'world':
            sp = datum['world']['sp']
            ap = datum['world']['ap']

            audio = pw.synthesize(pitch, sp, ap, SAMPLE_RATE)
        else:
            pitch_at_sr = to_sample_rate(pitch)

            # Pick oscillator.
            if synth_engine == 'blit':
                audio = blit(pitch_at_sr)
            elif synth_engine == 'additive':
                audio = additive(pitch_at_sr)

            # Filter with average AR envelope.
            audio = lfilter([1], average_coefficients, audio)

        audio = remove_dc(audio)
        audio = normalize(audio)

        if audio_fade:
            audio = add_fade(audio, audio_fade, rate=SAMPLE_RATE)

        filename = "proc_" + datum['filename']
        write_path = os.path.join(PROCESSED_PATH, filename)

        print("Writing {}...".format(filename))
        wavfile.write(write_path, SAMPLE_RATE, audio)
