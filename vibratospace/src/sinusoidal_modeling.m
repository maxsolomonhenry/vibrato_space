% Rather nasty work-around
PICKLE_PATH = '/Users/maxsolomonhenry/Documents/Python/vibrato_space/vibratospace/data/data.pickle';
SM_PATH = '/Users/maxsolomonhenry/Documents/MATLAB/sinusoidal-model';

% Load SM library.
addpath(genpath(SM_PATH));

% Load pickle.
fid = py.open(PICKLE_PATH, 'rb');
data = py.pickle.load(fid);

hold on;

for k = 1:length(data)
	temp = data{k}{'frequency'};
	plot(temp);
end

hold off;


% Hann analysis window
winflag = 3;

% Display name of analysis window in the terminal
fprintf(1,'%s analysis window\n',infowin(winflag,'name'));

% Flag for center of first window
cfwflag = {'nhalf','one','half'};
cf = 1;

% Flag for log magnitude spectrum
lmsflag = {'dbr','dbp','nep','oct','bel'};
lmsf = 2;

% Magnitude spectrum scaling
magflag = {'nne','lin','log','pow'};
mf = 3;

% Number of fundamental periods
nT0 = 3;

% Normalize analysis window
normflag = true;

% Use zero phase window
zphflag = true;

% Replace -Inf in spectrogram
nanflag = false;

% Maximum number of peaks to retrieve from analysis
maxnpeak = 80;

% Relative threshold
%relthres = -inf(1);
relthres = -100;

% Absolute threshold
%absthres = -inf(1);
absthres = -120;

% Partial tracking flag
ptrackflag = {'','p2p'};
ptrck = 2;

% Resynthesis flag
synthflag = {'OLA','PI','PRFI'};
rf = 2;

% Display flag
dispflag = true;
