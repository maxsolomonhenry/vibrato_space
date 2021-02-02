from vibratospace.src.defaults import PICKLE_PATH
from vibratospace.src.util import load_data
import matlab.engine

data = load_data(PICKLE_PATH)

# This script uses Marcelo Caetano's sinusoidal model (for Matlab).
# https://github.com/marcelo-caetano/sinusoidal-model

SM_PATH = '/Users/maxsolomonhenry/Documents/MATLAB/sinusoidal-model'

# Load Matlab. ...I know...
eng = matlab.engine.start_matlab()

# Add
temp = eng.genpath(SM_PATH)
eng.addpath(temp)

eng.run_sm(nargout=0)
