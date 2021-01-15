import pickle
import matplotlib.pyplot as plt
import crepe
import os

from scipy.io import wavfile
from scipy.signal import lfilter
from src.defaults import PICKLE_PATH, FIG_PATH, AUDIO_TEST_PATH
from src.oscillators import AdditiveOsc, Blit
from src.util import *

with open(PICKLE_PATH, 'rb') as handle:
    data = pickle.load(handle)

# Grab an arbitrary sound sample.
datum = data[int(np.round(np.random.rand()*len(data)))]

fig_filename = "fig__" + os.path.splitext(datum['filename'])[0] + ".pdf"
fig_file_path = os.path.join(FIG_PATH, fig_filename)

exc_filename = "exc__" + datum['filename']
exc_file_path = os.path.join(AUDIO_TEST_PATH, exc_filename)

print("Writing {}...".format(exc_filename))

wavfile.write(
    exc_file_path,
    SAMPLE_RATE,
    add_fade(datum['audio'], 0.125, rate=SAMPLE_RATE)
)


# Set up synthesis.
blit = Blit()
additive = AdditiveOsc(num_harmonics=25)

oscs = [
    {'osc': blit, 'name': 'blit'},
    {'osc': additive, 'name': 'add'}
]

# Set up plots.
t = np.linspace(
    0,
    len(datum['frequency']) / float(PITCH_RATE),
    len(datum['frequency']),
    endpoint=False
)

plt.plot(t, datum['frequency'], label='orig')

for osc in oscs:

    pitch = datum['frequency']
    # pitch = hz_to_midi(pitch)
    # pitch = repitch(pitch, 48)
    # pitch = add_fade(pitch, 0.125, rate=PITCH_RATE)

    # pitch = fix_deviation(pitch, pitch_deviation)
    # pitch = midi_to_hz(pitch)
    pitch = to_sample_rate(pitch)

    audio = osc['osc'](pitch)
    audio = lfilter([1], datum['lpc'], audio)

    audio = normalize(audio)
    audio = add_fade(audio, 0.125, rate=SAMPLE_RATE)

    syn_filename = osc['name'] + "__" + datum['filename']
    syn_file_path = os.path.join(AUDIO_TEST_PATH, syn_filename)

    print("Writing {}...".format(syn_filename))
    wavfile.write(syn_file_path, SAMPLE_RATE, audio)

    frequency = crepe.predict(
        audio,
        SAMPLE_RATE,
        step_size=1/PITCH_RATE * 1000,
        viterbi=True,
    )[1]

    if len(frequency) > len(t):
        frequency = frequency[:len(t)]

    plt.plot(t, frequency, label=osc['name'])

plt.legend()
plt.title(datum['filename'])
plt.xlabel('time (s)')
plt.ylabel('frequency (Hz)')
plt.savefig(fig_file_path)
