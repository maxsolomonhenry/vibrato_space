import numpy as np
from vibratospace.src.defaults import SAMPLE_RATE, EPS
from vibratospace.src.util import normalize, fix_length


class Blit:
    """Band-limited impulse train.

    STK's BLIT class ported from C++ and slightly modified. Thanks Gary.
    """

    def __init__(self):
        self.phase = 0
        self.output = []

    def __call__(self, hz: np.ndarray) -> np.ndarray:
        """
        Accepts sample-rate array of fundamental frequency to re-synthesize.
        """
        self.reset()

        for frequency in hz:
            p_, rate_ = self.set_frequency(frequency)
            m_ = self.update_harmonics(p_)

            denominator = np.sin(np.pi * self.phase)

            if denominator <= EPS:
                temp = 1
            else:
                temp = np.sin(m_ * np.pi * self.phase)
                temp /= m_ * denominator

            self.phase += rate_

            if self.phase >= 1:
                self.phase -= 1

            self.output.append(temp)
        return np.array(self.output) - np.mean(self.output)

    def reset(self):
        self.phase = 0
        self.output = []

    @staticmethod
    def set_frequency(frequency):
        p_ = SAMPLE_RATE / frequency
        rate_ = 1 / p_
        return p_, rate_

    @staticmethod
    def update_harmonics(p_):
        max_harmonics = np.floor(0.5 * p_)
        return 2 * max_harmonics + 1


class AdditiveOsc:
    """
    Oscillator that adds a given number of harmonics. Call accepts an array of
    instantaneous frequency, outputs audio.

    Uses technique to account for accumulation error. With apologies to:

    Abe, Toshihiko, and Masaaki Honda. "Sinusoidal model based on instantaneous
        frequency attractors." IEEE Transactions on Audio, Speech, and Language
        Processing 14.4 (2006): 1292-1300.
    """

    def __init__(self, num_harmonics: int, chunk_size: int = 1000):
        assert 0 < num_harmonics, 'Must specify a positive number of harmonics.'
        self.num_harmonics = num_harmonics

        assert chunk_size > 50, "Chunk size must be greater than 50 samples."
        self.chunk_size = chunk_size

    def __call__(self, hz: np.ndarray) -> np.ndarray:
        phase = self.hz_to_phase(hz)
        x = 0
        for h in np.arange(1, self.num_harmonics + 1):
            init_phase = np.random.rand() * 2 * np.pi
            x += np.cos(h * phase + init_phase)
        return normalize(x)

    def hz_to_phase(self, hz: np.ndarray) -> np.ndarray:
        """ Cumulative sum that side-steps accumulation error.

        A simple version of this function goes as follows:

        np.cumsum(2 * np.pi * hz / SAMPLE_RATE)

        I.e., accumulate instantaneous frequency (to generate phase.) The method
        below does this calculation in chunks, to avoid accumulation error.
        """

        hz = 2 * np.pi * hz / SAMPLE_RATE

        # Condition array for chunking.
        remainder = len(hz) % self.chunk_size
        if remainder:
            hz = fix_length(hz, len(hz) + (self.chunk_size - remainder))

        # Calculate phase in small chunks (less accumulated error).
        num_chunks = len(hz) // self.chunk_size
        chunks = np.reshape(hz, [num_chunks, self.chunk_size])
        phase = np.cumsum(chunks, axis=1)

        # Add offset to smooth phase trajectory.
        offset = phase[:, -1][:-1]
        offset = np.pad(offset, (1, 0))
        offset = np.cumsum(offset) % (2.0 * np.pi)
        phase = (phase.T + offset).T
        phase = phase.reshape(len(hz))
        phase = phase % (2.0 * np.pi)

        return phase
