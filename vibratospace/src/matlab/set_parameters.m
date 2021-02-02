%
% Prelimiary script for sinusoidal-model settings.
%

% Hann analysis window
winflag = 3;

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
normflag = false;

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

% Assume vibrato frequency of 6 Hz.
f0 = 6;

% Frame size = n*T0
[framelen,ref0] = framesize(f0,fs,nT0);

% 50% overlap
hop = hopsize(framelen,0.5);

% FFT size
nfft = fftsize(framelen);

% Frequency difference for peak matching (Hz)
delta = fix(ref0/2);