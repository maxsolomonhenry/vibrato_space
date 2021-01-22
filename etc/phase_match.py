"""
Reconstruct phase from pitch track using cubic interpolation.

Riffing on:

R. McAulay and T. Quatieri, “Speech analysis/synthesis based on a sinusoidal
    representation,” IEEE Trans. Acoust., Speech, Signal Process., vol. 34,
    no. 4, pp. 744–754, Aug. 1986, doi: 10.1109/TASSP.1986.1164910.


TODO: broken.
"""

import numpy as np
from vibratospace.src.util import load_data, time_plot
from vibratospace.src.defaults import PICKLE_PATH, SAMPLE_RATE, PITCH_RATE


class PhaseInterp:
    """
    Phase from an array of instantaneous frequency using cubic interpolation.
    TODO: move to here.
    """
    pass


def ideal_M(phi_k: float,       # Starting phase.
            omega_k: float,     # Starting frequency (radians per second).
            phi_k1: float,      # Target phase.
            omega_k1: float,    # Target frequency.
            samp_period: float  # Sample period in seconds (1 / sr).
            ) -> int:
    """
    Finds most direct path for phase. M is an integer coefficient of 2 * pi.
    Eq. 36 in McCaulay & Quatieri 1986.
    """
    a = phi_k + omega_k * samp_period - phi_k1
    b = (omega_k1 - omega_k) * (samp_period / 2)
    return int(np.round((1/(2*np.pi)) * (a + b)))


def get_alpha_beta(phi_k, omega_k, phi_k1, omega_k1, M, samp_period):
    """
    Solving for first and second order phase coefficients.
    Eq. 34 in McCaulay & Quatieri 1986.
    """
    A = np.array([[3/samp_period**2,    -1/samp_period],
                  [-2/samp_period**3, 1/samp_period**2]])
    B = np.array([phi_k1 - phi_k - omega_k*samp_period + (2 * np.pi * M),
                  omega_k1 - omega_k])
    return np.matmul(A, B)


def predict_phase(phi_k, omega_k1, samp_period):
    """
    Predicts a target phase value for interpolation.
    """
    return (phi_k + omega_k1 * samp_period) % (2 * np.pi)


"""
    t = np.arange(0, T, 1/SAMPLE_RATE)
    phase = phi_k + omega_k * t + alpha * t **2 + beta * t ** 3
"""


if __name__ == '__main__':
    data = load_data(PICKLE_PATH)

    # Work from one example for now.
    datum = data[0]

    hz = datum['frequency']

    pitch_period = 1/PITCH_RATE
    sample_period = 1/SAMPLE_RATE
    t = np.arange(0, pitch_period, sample_period)

    # TODO: Assumes initial frame of zero. Maybe a problem.
    phase_trajectory = np.array([])
    target_omega = 0
    target_phase = 0

    # Assumes 0th angular speed is the same.
    start_omega = hz[0]

    for frequency in hz:
        start_phase = target_phase  # % (2 * np.pi)
        target_omega = (2 * np.pi) * frequency
        target_phase = predict_phase(start_phase, target_omega, pitch_period)

        # Find ideal phase target within the class of  mod 2pi.
        M = ideal_M(start_phase, start_omega, target_phase, target_omega,
                    pitch_period)

        alpha, beta = get_alpha_beta(start_phase, start_omega, target_phase,
                                     target_omega, M, pitch_period)

        temp = \
            start_phase + start_omega * t + alpha * t ** 2 + beta * t ** 3

        phase_trajectory = np.append(phase_trajectory, temp)

    time_plot(phase_trajectory, show=False, rate=SAMPLE_RATE)
    x = np.cos(phase_trajectory)
    time_plot(x, rate=SAMPLE_RATE)
