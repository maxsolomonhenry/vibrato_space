"""
Reads .pickle file containing WORLD analysis data, manipulates signals, and
re-synthesizes.

M. Morise, F. Yokomori, and K. Ozawa, ``WORLD: a vocoder-based high-quality
    speech synthesis system for real-time applications,'' IEICE transactions on
    information and systems, vol. E99-D, no. 7, pp. 1877-1884, 2016.
"""

import os
import matlab.engine
import pyworld as pw
from scipy.io import wavfile

from vibratospace.src.python.defaults import (
    AUDIO_TEST_PATH,
    DATA_PATH,
    MATLAB_SCRIPT_PATH,
    PITCH_RATE,
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
    repitch
)


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

    # Load analysis data.
    pickle_path = os.path.join(DATA_PATH, 'world_data.pickle')
    pickle_path = os.path.abspath(pickle_path)
    data = load_data(pickle_path)

    # Start Matlab engine (gross, I know).
    print('Starting Matlab engine...')
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.realpath(MATLAB_SCRIPT_PATH))

    # Synthesis parameters.
    repitch_note = None  # 50
    audio_fade = .25
    pitch_std = None  # 10
    num_partials = 1

    for datum in data:
        filename = datum['filename']
        pitch = datum['frequency']
        sp = datum['sp']

        ap = datum['ap']
        sample_rate = datum['sample_rate']

        pitch = hz_to_midi(pitch)
        pitch = sine_model_resynth(pitch, PITCH_RATE, num_partials, SM_PATH)

        # Label filename with synthesis parameters.
        file_prefix = "proc__"

        # Synthesis flags.
        if repitch_note:
            pitch = repitch(pitch, repitch_note)
            file_prefix += "repitch_"
        if pitch_std:
            pitch = fix_deviation(pitch, pitch_std)
            file_prefix += "std_"

        pitch = midi_to_hz(pitch)

        # Re-synthesize with WORLD.
        audio = pw.synthesize(pitch, sp, ap, sample_rate)
        audio = remove_dc(audio)
        audio = normalize(audio)

        if audio_fade:
            audio = add_fade(audio, audio_fade, rate=sample_rate)
            file_prefix += "audfade_"

        synthesis_filename = file_prefix + filename
        print('Writing {}...'.format(synthesis_filename))
        write_path = os.path.join(AUDIO_TEST_PATH, synthesis_filename)
        wavfile.write(write_path, sample_rate, audio)
