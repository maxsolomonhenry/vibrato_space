function [sinusoidal, srer] = sineModelResynthesis(wav, fs, num_partials, SM_PATH)
%
%   Convenience wrapper for Marcelo Caetano's sinusoidal model, that allows 
%   for resynthesis with a fixed number of partials, as specified by
%   `num_partials`. Settings are otherwise default.
%
%   For Marcelo's script, see:
%
%   https://github.com/marcelo-caetano/sinusoidal-model/
%

if nargin < 3
    num_partials = [];
end

if nargin < 4
    SM_PATH = '/Users/maxsolomonhenry/Documents/MATLAB/sinusoidal-model';
end

% If first run, initlaize sinusoidal model variables/parameters.
if ~exist('winflag', 'var')
    addpath(genpath(SM_PATH));
    run('set_parameters.m');
end

if size(wav, 1) < size(wav, 2)
    wav = wav';
end

if size(wav, 2) ~= 1
    error('Error. More than one channel detected. Mono files only please.');
end

mean_ = mean(wav);
wav = wav - mean_;

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

% Restrict resynthesis to specified number of partials.
if ~isempty(num_partials)
    if num_partials > size(amplitude, 1)
        warning(...
            'Requested partials (%d) is greater than detected paths (%d).',...
            num_partials, size(amplitude, 1)...
        );
    
        % Fix to maximum number of partials.
        num_partials = size(amplitude, 1);
    end
    
    temp = NaN(size(amplitude));
    temp(1:num_partials, :) = amplitude(1:num_partials, :);
    amplitude = temp;
end

[sinusoidal,partial,amp_partial,freq_partial,phase_partial] = ...
    sinusoidal_resynthesis(...
        amplitude,frequency,phase,...
        framelen,...
        hop,...
        fs,...
        nsample,...
        center_frame,...
        npartial,...
        nframe,...
        delta,...
        winflag,...
        cfwflag{cf},...
        synthflag{rf},...
        ptrackflag{ptrck},...
        dispflag...
    );

residual = wav - sinusoidal;

% Calculate signal to resynthesis energy ratio (SRER)
srer = lin2log(std(wav)/std(residual),'dbp');
sinusoidal = sinusoidal + mean_;

end