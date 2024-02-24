""" """

import numpy as np


def asymmetric_lorentzian(f, ofs, height, phi, fr, fwhm):
    """
    Lorentzian fit fn for resonator magnitude response in logarithmic scale
    f: array of probe frequencies (independent variables)
    ofs: y offset
    height: Lorentzian peak/dip to floor distance
    phi: phase factor to account for peak asymmetry
    fr: resonant frequency (peak/dip location)
    fwhm: full width half maxmimum of peak/dip
    """
    numerator = height * np.exp(1j * phi)
    denominator = 1 + 2j * ((f - fr) / fwhm)
    return 20 * np.log10(np.abs(1 - numerator / denominator)) + ofs
