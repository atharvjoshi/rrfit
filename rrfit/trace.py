""" """

import numpy as np

from rrfit.calibrations import fit_cable_delay, fit_background


class Trace:
    """
    Encapsulates an S21 trace and its acquisition and analysis record

    the trace raw data:
    an array of independent variables called f
    two arrays of dependent variables - real and imag (optional) parts

    **kwargs: include attributes such as power, temp, fitted params
    """

    def __init__(self, f, real, imag, **kwargs):
        """ """

        # independent variable
        self.f = f

        # dependent variables
        self._real = real
        self._imag = imag

        # control variables
        self.power = None
        self.temperature = None

        # calibration parameters
        self.cable_delay = None
        self.orp = None  # off-resonant point

        # fit parameters

        for name, value in kwargs.items():
            setattr(self, name, value)

    def s21raw(self, sign=1):
        """
        sign must be 1 or -1
        """
        return self._real + sign * 1j * self._imag

    def s21nodelay(self, cable_delay=None):
        """s21 with cable delay removed, with an option to set a manual cable delay override"""
        if cable_delay is None:
            cable_delay = self.cable_delay
        return self.s21raw() * np.exp(-1j * 2 * np.pi * self.f * cable_delay)

    def s21canonical(self, orp=None):
        """s21 at the canonical position, with an option to set a manual off resonant point override"""
        if orp is None:
            orp = self.orp
        return self.s21nodelay() / orp

    def remove_cable_delay(self, exclude=None):
        """ """
        s21phase = np.unwrap(np.angle(self.s21raw()))
        self.cable_delay = fit_cable_delay(s21phase, self.f, exclude=exclude)
        return self.s21nodelay()

    def remove_background(self):
        """ """
        self.orp = fit_background(self.s21nodelay(), self.f)
        return self.s21canonical()

    def fit_s21(self):
        """ """
        # TODO
