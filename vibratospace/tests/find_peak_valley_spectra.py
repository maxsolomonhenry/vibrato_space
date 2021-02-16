import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from vibratospace.src.python.defaults import DATA_PATH, PITCH_RATE, SAMPLE_RATE
from vibratospace.src.python.util import load_data, time_plot, to_sample_rate, high_pass, save_data


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


if __name__ == '__main__':
    PLOTS = True

    data = load_data(
        os.path.join(DATA_PATH, 'data.pickle')
    )

    spectra = []

    for k in range(1):
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
            valley_f0 = f0[valleys[which_peak + 1]]
            mid_f0 = (peak_f0 * valley_f0) ** (1/2)

            peak_ = peaks[which_peak] / PITCH_RATE * SAMPLE_RATE
            valley_ = valleys[which_peak + 1] / PITCH_RATE * SAMPLE_RATE
            mid_ = np.mean([peak_, valley_])

            peak_ = int(round(peak_))
            valley_ = int(round(valley_))
            mid_ = int(round(mid_))

            half_peak_length = int(3 * (1/peak_f0) * SAMPLE_RATE)
            half_valley_length = int(3 * (1/valley_f0) * SAMPLE_RATE)
            half_mid_length = int(3 * (1 / mid_f0) * SAMPLE_RATE)

            # half_peak_length = 4096
            # half_valley_length = 4096
            # half_mid_length = 4096

            peak_idx = np.arange(-half_peak_length, half_peak_length + 1, dtype=int) + peak_
            valley_idx = np.arange(-half_valley_length, half_valley_length + 1,
                                     dtype=int) + valley_
            mid_idx = np.arange(-half_mid_length, half_mid_length + 1, dtype=int) + mid_

            peak_frame = datum['audio'][peak_idx] * np.hanning(peak_idx.shape[0])
            valley_frame = datum['audio'][valley_idx] * np.hanning(valley_idx.shape[0])
            mid_frame = datum['audio'][mid_idx] * np.hanning(mid_idx.shape[0])

            spec_peak = np.fft.fft(peak_frame, next_power_of_2(len(peak_frame)))
            spec_valley = np.fft.fft(valley_frame, next_power_of_2(len(valley_frame)))
            spec_mid = np.fft.fft(mid_frame, next_power_of_2(len(mid_frame)))

            spectra.append(
                {
                    'filename': datum['filename'],
                    'peak': spec_peak,
                    'mid': spec_mid,
                    'valley': spec_valley
                }
            )

            if PLOTS:
                plt.subplot(6, 1, which_peak + 1)
                time_plot(np.abs(spec_peak[:500]), title=datum['filename'], show=False)
                time_plot(np.abs(spec_mid[:500]), title=datum['filename'], show=False)
                time_plot(np.abs(spec_valley[:500]), show=False)

        if PLOTS:
            plt.show()

    path = os.path.join(DATA_PATH, 'peak_valley_spectra.pickle')
    save_data(path, spectra, force=True)
