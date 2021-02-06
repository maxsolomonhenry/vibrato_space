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
axes_lbl.ttl = filename;

% Axes limits
axes_lim.tlim = [plot_wav.time(1) plot_wav.time(end)];
axes_lim.alim = [min(plot_wav.wav,[],'all','omitnan') max(plot_wav.wav,[],'all','omitnan')];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT SINUSOIDAL MODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make figure
mkfigwav(plot_wav,axes_lim,axes_lbl,fig_layout);