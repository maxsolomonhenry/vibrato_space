"""
    TODO: REMOVE ALPHA.
    The plan:

    Build a filter that takes two spectral envelopes of size N, and interpolates
    between them.

    The object is initialized with two spectral envelopes. At call, it accepts:
    (1) a modulation signal [0, 1] for interpolating between the two envelopes

    A separate object will filter via fast convolution:

    (1) Env to impulse response. (TODO: Minimum phase? Liftering?)
    (2) Taper to N samples with decaying exponential, pad to 2N.
    (3) Multiply in freq domain
    (4) Inv, then OLA.
"""

import copy
import numpy as np
from vibratospace.src.python.util import fix_length


class EnvelopeInterpolator:
    """
    The object is initialized with three spectral envelopes. At call, it accepts
    a modulation signal [-1, 1] for interpolating between the three envelopes,
    where -1 => env1, 0 => env2, and 1 => env3.

    `alpha` determines the span of influence of the middle spectrum:

    1   - linear cross-fading
    <1  - middle spectrum has more influence
    >1  - middle spectrum has less influence
    """
    def __init__(
            self,
            env1: np.ndarray,
            env2: np.ndarray,
            env3: np.ndarray = None,
            alpha: float = 1
    ):
        assert (
                env1.ndim == env2.ndim == 1
        ), 'Envelopes must be 1d arrays.'
        assert (
                env1.size == env2.size
        ), 'Envelopes must be the same length.'
        assert alpha > 0, 'Positive alpha values only.'

        if env3 is not None:
            assert (env3.ndim == 1 and env3.size == env1.size)

        self.env1 = env1
        self.env2 = env2
        self.env3 = env3
        self.alpha = alpha

    def __call__(
            self,
            modulator: np.ndarray,
            emphasis: float = 1.0
    ) -> np.ndarray:
        """
        Note: call accepts modulator signal at, presumably, frame-rate.
        """
        assert modulator.ndim == 1
        assert -1 <= modulator.all() <= 1
        assert emphasis > 0

        num_bins = self.env1.shape[0]
        num_frames = modulator.shape[0]
        modulator *= emphasis

        out_ = np.zeros([num_bins, num_frames])

        for f, val in enumerate(modulator):
            out_[:, f] = self.interp_arrays(
                self.env1,
                self.env2,
                self.env3,
                val
            )
        return np.maximum(0, out_)

    def interp_arrays(self, array1, array2, array3, val):
        # c1 = np.maximum(0, 1 - (val + 1) ** self.alpha)
        # c2 = (1 - np.abs(val)) ** self.alpha
        # c3 = np.maximum(0, 1 - (1 - val) ** self.alpha)
        if self.env3 is not None:
            c1 = np.maximum(0, -val)
            c2 = 1 - np.abs(val)
            c3 = np.maximum(0, val)
            answer = c1 * array1 + c2 * array2 + c3 * array3
        else:
            c1 = (1 - val)/2 * array1
            c2 = (val + 1)/2 * array2
            answer = c1 * array1 + c2 * array2

        return answer


class FastConvolve:
    """
    A separate object will filter via fast convolution:

    (1) Env to impulse response. (TODO: Minimum phase? Liftering?)
    (2) Taper to N samples with decaying exponential, pad to 2N.
    (3) Multiply in freq domain
    (4) Inv, then OLA.
    """
    def __init__(
            self,
            frame_size: int = 4096,
            hop_size: int = 512,
            debug: bool = False
    ):
        assert frame_size >= 32
        assert hop_size <= frame_size

        self.frame_size = frame_size
        self.hop_size = hop_size
        self.debug = debug

    def __call__(
            self,
            signal: np.ndarray,
            envelopes: np.ndarray
    ) -> np.ndarray:

        assert signal.ndim == 1
        assert envelopes.ndim == 2

        # Transpose envelopes array if necessary.
        if envelopes.shape[1] == self.frame_size:
            envelopes = envelopes.T

        # Ensure envelopes have appropriate frame size.
        assert envelopes.shape[0] == self.frame_size

        signal = self.pad_or_truncate_signal(signal, envelopes)
        out_ = self.apply_envelopes(signal, envelopes)

        return out_

    def apply_envelopes(self, in_, envelopes) -> np.ndarray:
        # Add extra 1.5 `frame_size` for ringing filter and padding.
        out_length = np.int(
            np.ceil(
                self.get_ola_length(envelopes) + self.frame_size
            )
        )
        out_ = np.zeros(out_length)
        in_ = np.pad(in_, self.frame_size // 2)

        # window = np.hamming(self.frame_size)
        impulse_taper = self.decaying_exponential(t0=self.frame_size, eps=1e-8)

        read_in = 0
        read_out = read_in + self.frame_size

        write_in = 0
        write_out = write_in + (2 * self.frame_size)

        for envelope in envelopes.T:
            x = copy.copy(in_[read_in:read_out])
            # x *= window
            X = np.fft.fft(x, self.frame_size * 2)

            envelope = self.condition_envelope(envelope, impulse_taper)

            temp = X * envelope
            temp = np.real(
                np.fft.ifft(temp)
            )

            out_[write_in:write_out] += temp

            if self.debug:
                plt.subplot(3, 1, 1)
                plt.title('input frame')
                plt.plot(x)
                plt.subplot(3, 1, 2)
                plt.title('output frame')
                plt.plot(temp)
                plt.subplot(3, 1, 3)
                plt.title('running output')
                plt.plot(out_)
                plt.show()

            # Increment pointers.
            read_in += self.hop_size
            read_out = read_in + self.frame_size

            write_in += self.hop_size
            write_out = write_in + (2 * self.frame_size)

        # TODO: Something like this. Do the math.
        # out_ = out_[self.frame_size:-self.frame_size]

        return out_

    def condition_envelope(self, envelope, impulse_taper):
        """
        Condition N-length spectral envelope for fast convolution.

        (1) Ifft to get impulse response.
        (2) Taper response with a decaying exponential.
        (3) Zero-pad to 2N and FFT to get `nice' spectrum.

        Note: returns 2N-length signal.
        """

        impulse_response = np.real(np.fft.ifft(envelope))
        temp = impulse_response * impulse_taper
        return np.fft.fft(temp, 2 * self.frame_size)

    def pad_or_truncate_signal(self, signal, envelopes):
        """
        Ensures that signal-length matches the number of envelopes provided.

        This assumes that there is one envelope per frame of the OLA
        reconstruction. The `signal` is truncated or padded, as required.
        """

        ola_length = self.get_ola_length(envelopes)

        signal = fix_length(signal, ola_length)

        return signal

    def get_ola_length(self, envelopes):
        num_frames = envelopes.shape[1]
        return (num_frames - 1) * self.hop_size + self.frame_size

    @staticmethod
    def decaying_exponential(t0: int, eps: float) -> np.ndarray:
        """
        Return exponential that decays from 1 to `eps` by index `t0`.
        """
        tau = -t0 / np.log(eps)
        return np.exp(-np.arange(t0)/tau)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Globals, parameters.
    SAMPLE_RATE = 44100
    num_bins = 1024
    hop_size = num_bins // 4

    # Test EnvelopeInterpolator
    test1 = np.zeros(num_bins)
    test2 = np.ones(num_bins)

    my_interp = EnvelopeInterpolator(test1, test2)

    # 180 frames @ 512 samples per hop and 44100 Hz sample-rate => ~2 sec audio.
    test_mod = np.cos(2 * np.pi * 2 * np.linspace(0, 2, 180))

    envelopes = my_interp(test_mod)
    plt.imshow(envelopes, origin='lower', aspect='auto')
    plt.show()

    # Test FastConvolve
    f0 = 440
    test_signal = np.cos(2 * np.pi * f0 * np.arange(0, 2, 1/SAMPLE_RATE))
    plt.plot(test_signal)
    plt.show()

    my_convolve = FastConvolve(
        frame_size=num_bins,
        hop_size=hop_size,
        debug=False
    )
    out_ = my_convolve(test_signal, envelopes)
    plt.plot(out_)
    plt.show()
