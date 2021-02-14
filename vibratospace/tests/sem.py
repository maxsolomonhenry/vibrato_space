import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from vibratospace.src.python.defaults import DATA_PATH, PITCH_RATE, SAMPLE_RATE
from vibratospace.src.python.util import load_data, time_plot, to_sample_rate, high_pass


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


if __name__ == '__main__':
    data = load_data(
        os.path.join(DATA_PATH, 'data.pickle')
    )

    for k in range(0, 15, 5):
        datum = data[k]
        f0 = datum['crepe_f0']
        mean_ = np.mean(f0)
        f0 -= mean_
        f0 -= high_pass(f0, 10)

        peaks, _ = find_peaks(f0)
        valleys, _ = find_peaks(-f0)

        f0 += mean_

        for which_peak in range(6):
            peak_f0 = f0[peaks[which_peak]]
            valley_f0 = f0[valleys[which_peak]]
            mid_f0 = (peak_f0 * valley_f0) ** 1/2

            peak_ = peaks[which_peak] / PITCH_RATE * SAMPLE_RATE
            valley_ = valleys[which_peak] / PITCH_RATE * SAMPLE_RATE
            mid_ = np.mean([peak_, valley_])

            peak_ = int(round(peak_))
            valley_ = int(round(valley_))
            mid_ = int(round(mid_))

            half_peak_length = int(1.5 * (1/peak_f0) * SAMPLE_RATE)
            half_valley_length = int(1.5 * (1/valley_f0) * SAMPLE_RATE)

            peak_idx = np.linspace(-half_peak_length, half_peak_length, dtype=int) + peak_
            valley_idx = np.linspace(-half_valley_length, half_valley_length, dtype=int) + valley_

            peak_frame = datum['audio'][peak_idx] * np.hanning(peak_idx.shape[0])
            valley_frame = datum['audio'][valley_idx] * np.hanning(valley_idx.shape[0])

            spec_peak = np.fft.fft(peak_frame, next_power_of_2(len(peak_frame) * 4))
            spec_valley = np.fft.fft(valley_frame, next_power_of_2(len(valley_frame) * 4))

            time_plot(np.abs(spec_peak), title=datum['filename'], show=False)
            time_plot(np.abs(spec_valley))

    # upsampled_f0 = to_sample_rate(f0)
    #
    # # time_plot(datum['audio'], SAMPLE_RATE, title=datum['filename'], show=False)
    # time_plot(upsampled_f0, SAMPLE_RATE, show=False)
    # plt.scatter(peaks, np.zeros(peaks.shape))
    # plt.scatter(valleys, np.zeros(valleys.shape))
    # plt.show()