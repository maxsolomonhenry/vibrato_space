% Test the wrapper.
clearvars;

PATH_TO_DATA = '/Users/maxsolomonhenry/Documents/Python/vibrato_space/vibratospace/data/data.pickle';

fid = py.open(PATH_TO_DATA,'rb');
data = py.pickle.load(fid);

[fs, wav] = getWav(data, 6);

% Plot test and reconstruction.
plot(sineModelResynthesis(wav, fs, 100));
% hold on
% plot(t, wav);
xlabel('time (s)');
ylabel('amplitude');
% hold off;

function [fs, wav] = getWav(data, k)
    temp = data(k);
    wav = double(temp{1}{'crepe_f0'});
    fs = double(temp{1}{'sample_rate'});
end