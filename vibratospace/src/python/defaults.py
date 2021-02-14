"""
Globals.
"""

import os

# Workaround to have absolute paths, no matter where the module is called from.
here = os.path.dirname(__file__)

# Audio sample rate for synthesis.
SAMPLE_RATE = 44100

# CREPE sample rate in Hz.
PITCH_RATE = 200

# Small value to avoid log(0).
EPS = 1e-6

# Pickle filename for saving data.
DATA_PATH = os.path.realpath(
    os.path.join(here, '../../data/')
)

# Path for processed audio.
PROCESSED_PATH = os.path.realpath(
    os.path.join(here, '../../audio/processed/')
)

# Path for raw audio (to be analyzed, re-synthesized).
RAW_AUDIO_PATH = os.path.realpath(
    os.path.join(here, '../../audio/raw/**/*.wav')
)

# Path for figures.
FIG_PATH = os.path.realpath(
    os.path.join(here, '../../figs/')
)

# Path for audio test outputs.
AUDIO_TEST_PATH = os.path.realpath(
    os.path.join(here, '../../audio/tests/')
)

# Path to MATLAB scripts within this project.
MATLAB_SCRIPT_PATH = os.path.realpath(
    os.path.join(here, '../matlab/')
)

# Path for MATLAB sinusoidal model.
# http://github.com/marcelo-caetano/sinusoidal-model/
SM_PATH = os.path.realpath(
    os.path.join(
        here, '/Users/maxsolomonhenry/Documents/MATLAB/sinusoidal-model'
    )
)

