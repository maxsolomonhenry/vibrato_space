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

run('show_model_plots.m');

[sinusoidal,partial,amp_partial,freq_partial,phase_partial] = sinusoidal_resynthesis(amplitude,frequency,phase,...
    framelen,hop,fs,nsample,center_frame,npartial,nframe,delta,winflag,cfwflag{cf},synthflag{rf},ptrackflag{ptrck},dispflag);

% Make residual
residual = wav - sinusoidal;

% Calculate signal to resynthesis energy ratio (SRER)
srer = lin2log(std(wav)/std(residual),'dbp');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURE PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot data
plot_wav.time = mktime(nsample,fs);
plot_wav.wav = wav;
plot_wav.wav(:,2) = sinusoidal;
plot_wav.wav(:,3) = residual;

% Figure layout
fig_layout.font = 'Times New Roman';
fig_layout.axesfs = 14;
fig_layout.titlefs = 22;
fig_layout.bckgdc = [1 1 1];
fig_layout.cmap = 'gray';
fig_layout.figsize = [15 10];
fig_layout.figpos = [0.5 0.5 fig_layout.figsize-0.5];
fig_layout.figunit = 'centimeters';
fig_layout.linsty = '-'; %'none'
fig_layout.linwidth = 1;
fig_layout.disp = 'on';
fig_layout.print = 'opengl';
fig_layout.legdisp = ["Original";"Sinusoidal";"Residual"];

% Axes label
axes_lbl.tlbl = 'Time (s)';
axes_lbl.albl = 'Amplitude (Normalized)';
% axes_lbl.ttl = sprintf('SRER: %2.2fdB %s',srer,sndname);

% Axes limits
axes_lim.tlim = [plot_wav.time(1) plot_wav.time(end)];
axes_lim.alim = [min(plot_wav.wav,[],'all','omitnan') max(plot_wav.wav,[],'all','omitnan')];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT SINUSOIDAL MODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make figure
mkfigwav(plot_wav,axes_lim,axes_lbl,fig_layout);
