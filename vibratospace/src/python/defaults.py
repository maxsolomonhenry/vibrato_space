"""
Globals.
"""

# Audio sample rate for synthesis.
SAMPLE_RATE = 44100

# CREPE sample rate in Hz.
PITCH_RATE = 100

# Small value to avoid log(0).
EPS = 1e-6

# Pickle filename for saving data.
DATA_PATH = '../../data/'

# Path for processed audio.
PROCESSED_PATH = '../../audio/processed/'

# Path for raw audio (to be analyzed, re-synthesized).
RAW_AUDIO_PATH = '../../audio/raw/**/*.wav'

# Path for figures.
FIG_PATH = '../../figs/'

# Path for audio test outputs.
AUDIO_TEST_PATH = '../../audio/tests/'

# Path to MATLAB scripts within this project.
MATLAB_SCRIPT_PATH = '../matlab/'

# Path for MATLAB sinusoidal model.
# http://github.com/marcelo-caetano/sinusoidal-model/
SM_PATH = '/Users/maxsolomonhenry/Documents/MATLAB/sinusoidal-model'

