% TODO: Refactor to inherit these from Python (when calling script?).
PICKLE_PATH = '/Users/maxsolomonhenry/Documents/Python/vibrato_space/vibratospace/data/data.pickle';
SM_PATH = '/Users/maxsolomonhenry/Documents/MATLAB/sinusoidal-model';
PITCH_RATE = 100;

% Load SM library.
addpath(genpath(SM_PATH));

% Load pickle.
fid = py.open(PICKLE_PATH, 'rb');
data = py.pickle.load(fid);

% Set up sinusoidal model.
fs = PITCH_RATE;
run('set_parameters.m');

% % Step through pitch trajectories.
% for k = 1:length(data)
% 	wav = data{k}{'frequency'};
% 	plot(temp);
% end

k = round(rand() * length(data));

% k = 10;

wav = data{k}{'frequency'}.double()';
wav = hz_to_midi(wav);

mean_ = mean(wav);
wav = wav - mean_;

plot(wav);
title(data{k}{'filename'}.string());

[amplitude,frequency,phase,nsample,ds,center_frame,npartial,nframe] = ...
    sinusoidal_analysis(...
        wav,...
        framelen,...
        hop,...
        nfft,...
        fs,...
        maxnpeak,...
        relthres,...
        absthres,...
        delta,...
        winflag,...
        cfwflag{cf},...
        normflag,...
        zphflag,...
        magflag{mf},...
        ptrackflag{ptrck}...
    );

run('show_plots.m');
