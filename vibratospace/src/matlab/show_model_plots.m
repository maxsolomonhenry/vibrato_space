%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURE PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Figure layout
fig_layout.font = 'Times New Roman';
fig_layout.axesfs = 14;
fig_layout.titlefs = 22;
fig_layout.bckgdc = [1 1 1];
fig_layout.cmap = 'jet';
fig_layout.figsize = [15 10];
fig_layout.figpos = [0.5 0.5 fig_layout.figsize-0.5];
fig_layout.figunit = 'centimeters';
fig_layout.msize = 7;
fig_layout.mtype = '.';
fig_layout.linsty = '-'; %'none'
fig_layout.meshsty = 'row'; %'both'
fig_layout.disp = 'on';
fig_layout.print = 'opengl';

% Axes label
axes_lbl.tlbl = 'Time (s)';
axes_lbl.flbl = 'Frequency (kHz)';
axes_lbl.dblbl = 'Spectral Energy (dB)';

% Axes limits
axes_lim.flim = [0 0.05];
axes_lim.dblim = [-120 0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT PARTIAL TRACKING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Log Mag Amp peaks
plot_part.specpeak = lin2log(amplitude,lmsflag{lmsf},nanflag);

% Time peaks
plot_part.time = repmat(center_frame/fs,[1,npartial])';

% Frequency peaks
plot_part.frequency = frequency/1000;

% Time limits
axes_lim.tlim = [plot_part.time(1,1) plot_part.time(1,end)];

% Title
axes_lbl.ttl = 'Partial Tracking';

% Make figure
mkfigpeakgram(plot_part,axes_lim,axes_lbl,fig_layout);
