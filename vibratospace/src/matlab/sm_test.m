% Test the wrapper.

PATH_TO_DATA = '/Users/maxsolomonhenry/Documents/Python/vibrato_space/vibratospace/data/world_data.pickle';

fid = py.open(PATH_TO_DATA,'rb');
data = py.pickle.load(fid);

wav = getWav(data, 1);

% Plot test and reconstruction.
plot(sineModelResynthesis(wav, fs, 5));
% hold on
% plot(t, wav);
xlabel('time (s)');
ylabel('amplitude');
% hold off;

function wav = getWav(data, k)
    temp = data(k);
    wav = double(temp{1}{'frequency'});
end