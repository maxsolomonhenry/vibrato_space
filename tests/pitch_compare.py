import pickle
import matplotlib.pyplot as plt
import crepe

from scipy.signal import lfilter
from src.defaults import PICKLE_PATH
from src.oscillators import AdditiveOsc, Blit
from src.util import *

with open(PICKLE_PATH, 'rb') as handle:
    data = pickle.load(handle)

datum = data[int(np.round(np.random.rand()*len(data)))]

blit = Blit()
additive = AdditiveOsc(num_harmonics=25)

plt.plot(datum['frequency'], label='orig')

oscs = [
    # {'osc': blit, 'name': 'blit'},
    {'osc': additive, 'name': 'add'}
]

for osc in oscs:

    pitch = datum['frequency']
    # pitch = hz_to_midi(pitch)
    # pitch = repitch(pitch, 48)
    # pitch = add_fade(pitch, 0.125, rate=PITCH_RATE)

    # pitch = fix_deviation(pitch, pitch_deviation)
    # pitch = midi_to_hz(pitch)
    pitch = to_sample_rate(pitch)

    audio = osc['osc'](pitch)
    # audio = lfilter([1], datum['lpc'], audio)

    audio = normalize(audio)
    # audio = add_fade(audio, 0.25, rate=SAMPLE_RATE)

    frequency = crepe.predict(
        audio,
        SAMPLE_RATE,
        step_size=1/PITCH_RATE * 1000,
        viterbi=True,
    )[1]

    plt.plot(frequency, label=osc['name'])

plt.legend()
plt.title(datum['filename'])
plt.show()
