import matplotlib.pyplot as plt
import numpy as np
import crepe
import os
from scipy.io import wavfile
from scipy.signal import lfilter, medfilt

from vibratospace.src.defaults import PICKLE_PATH, FIG_PATH, AUDIO_TEST_PATH, SAMPLE_RATE, PITCH_RATE
from vibratospace.src.oscillators import AdditiveOsc, Blit
from vibratospace.src.upsample import linear, quadratic, cubic, next_, prev_, nearest
from vibratospace.src.util import load_data, add_fade, to_sample_rate, normalize

data = load_data(PICKLE_PATH)

# Grab an arbitrary sound sample.
datum = data[int(np.round(np.random.rand()*len(data)))]

# Generate file names/paths for figure and sound tests.
fig_filename = "fig__" + os.path.splitext(datum['filename'])[0] + ".pdf"
fig_file_path = os.path.join(FIG_PATH, fig_filename)

exc_filename = "exc__" + datum['filename']
exc_file_path = os.path.join(AUDIO_TEST_PATH, exc_filename)

print("Writing {}...".format(exc_filename))

# Write unadulterated audio.
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

interps = [
    {'func': linear, 'name': 'linear'},
    # {'func': quadratic, 'name': 'quadratic'},
    # {'func': cubic, 'name': 'cubic'},
    {'func': next_, 'name': 'next'},
    {'func': prev_, 'name': 'previous'},
    {'func': nearest, 'name': 'nearest'},
    {'func': to_sample_rate, 'name': 'standard'}
]

# Set up plots.
t = np.linspace(
    0,
    len(datum['frequency']) / float(PITCH_RATE),
    len(datum['frequency']),
    endpoint=False
)

plt.plot(t, datum['frequency'], label='orig')

for interp in interps:
    for osc in oscs:

        pitch = datum['frequency']
        # pitch = medfilt(pitch, 5)
        # pitch = hz_to_midi(pitch)
        # pitch = repitch(pitch, 48)
        # pitch = add_fade(pitch, 0.125, rate=PITCH_RATE)

        # pitch = fix_deviation(pitch, pitch_deviation)
        # pitch = midi_to_hz(pitch)
        pitch = interp['func'](pitch)

        audio = osc['osc'](pitch)
        audio = lfilter([1], datum['lpc'], audio)

        audio = normalize(audio)
        audio = add_fade(audio, 0.125, rate=SAMPLE_RATE)

        syn_filename = interp['name'] + "_" + osc['name'] + "__" + datum['filename']
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

        plt.plot(t, frequency, label=interp['name'] + " " + osc['name'])

plt.legend()
plt.title(datum['filename'])
plt.xlabel('time (s)')
plt.ylabel('frequency (Hz)')
plt.savefig(fig_file_path)
