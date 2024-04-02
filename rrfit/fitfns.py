""" independent data argument must be called 'x' """

import numpy as np


def rr_s21(fs, nports, fr, Ql, Qc, phi=0, a=1, alpha=0, tau=0):
    """
    complex s21 model for resonators
    fs: array of probe frequencies (independent variables)
    nports: set to 1 for reflection and 2 for transmission measurements (constant)
    fr: resonance frequency
    Ql: loaded (total) quality factor
    Qc: coupling quality factor
    phi: phase due to circuit asymmetry
    a: net amplitude loss/gain due to measurement line
    alpha: net phase shift due to measurement line
    tau: frequency-dependent phase shift due to finite measurement line length
    """
    background = a * np.exp(-1j * (2 * np.pi * fs * tau + alpha))
    resonator = (Ql / Qc * np.exp(1j * phi)) / (1 + 2j * Ql * (fs / fr - 1))
    return background * (1 - (2 // nports) * resonator)


def asymmetric_lorentzian(x, ofs, height, phi, fr, fwhm):
    """
    Lorentzian fit fn for resonator magnitude response in logarithmic scale
    x: array of probe frequencies (independent variables)
    ofs: y offset
    height: Lorentzian peak/dip to floor distance
    phi: phase factor to account for peak asymmetry
    fr: resonant frequency (peak/dip location)
    fwhm: full width half maxmimum of peak/dip
    """
    numerator = height * np.exp(1j * phi)
    denominator = 1 + 2j * ((x - fr) / fwhm)
    return 20 * np.log10(np.abs(1 - numerator / denominator)) + ofs


def cable_delay_linear(x, tau, theta):
    """
    linear model which gives an initial guess for correcting cable delay tau
    fit this with the unwrapped phase of complex resonator signal
    x: array of probe frequencies (independent variables)
    tau: cable delay
    theta: arbitrary phase offset
    """
    return 2 * np.pi * x * tau + theta


def centered_phase(x, fr, Ql, theta, sign):
    """
    Arctan fit for resonator phase response around complex plane origin
    x: array of probe frequencies (independent variables)
    fr: resonant frequency
    Ql: loaded (total) quality factor
    theta: arbitrary ohase y-offset
    sign: whether
    note that np.arctan return real values in the interval [-pi/2, pi/2]
    """
    return theta + 2 * np.arctan(2 * Ql * (1 - x / fr))
