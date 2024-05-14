""" """

import numpy as np
import matplotlib.pyplot as plt
from uncertainties.umath import cos

from rrfit.calibrations import fit_cable_delay, fit_background
from rrfit.circlefit import fit_circle
from rrfit.fitfns import rr_s21_hanger
from rrfit.models import S21CenteredPhaseModel, S21Model


class Trace:
    """
    Encapsulates an S21 trace and its acquisition and analysis record

    the trace raw data:
    an array of independent variables called f
    two arrays of dependent variables - real and imag (optional) parts

    **kwargs: include attributes such as power, temp, fitted params
    """

    def __init__(self, id, f, real, imag, **kwargs):
        """ """
        self.id = id

        # independent variable
        self.f = f

        # dependent variables
        self._real = real
        self._imag = imag
        self.sign = 1

        # control variables
        self.power = None
        self.temperature = None

        # calibration parameters
        self.tau = None  # cable delay
        self.a = None  # background amplitude
        self.alpha = None  # background phase
        self.orp = None  # off-resonant point, complex number

        # fit parameters
        self.fr = None
        self.fr_err = None
        self.Ql = None
        self.Ql_err = None
        self.Qi = None
        self.Qi_err = None
        self.absQc = None
        self.absQc_err = None
        self.phi = None
        self.phi_err = None

        # other fit results
        self.fit_result = None
        self.best_fit = None
        self.residuals = None

        # toggle to exclude from final results
        self.is_excluded = False

        for name, value in kwargs.items():
            setattr(self, name, value)

    def s21raw(self):
        """
        sign must be 1 or -1  # TODO decide whether to make sign user-settable
        """
        return self._real + self.sign * 1j * self._imag

    def s21nodelay(self, cable_delay=None):
        """s21 with cable delay removed, with an option to set a manual cable delay override"""
        if cable_delay is None:
            cable_delay = self.tau
        return self.s21raw() * np.exp(-1j * 2 * np.pi * self.f * cable_delay)

    def s21canonical(self, orp=None):
        """s21 at the canonical position, with an option to set a manual off resonant point override"""
        if orp is None:
            orp = self.orp
        return self.s21nodelay() / orp

    def remove_cable_delay(self, exclude=None):
        """ """
        s21phase = np.unwrap(np.angle(self.s21raw()))
        self.tau = fit_cable_delay(s21phase, self.f, exclude=exclude)
        return self.s21nodelay()

    def remove_background(self, discont=None):
        """ """
        self.orp = fit_background(self.s21nodelay(), self.f, discont=discont)
        return self.s21canonical()

    def fit_s21(self, verbose=False, plot=False, discont=None):
        """ """
        # if cable delay hasn't been corrected yet, assume no cable delay
        if self.tau is None:
            self.tau = 0

        s21_canonical = self.remove_background(discont=discont)

        ## circle and phase fit to extract resonator parameters
        radius, center = fit_circle(s21_canonical)

        s21_centered_phase = np.unwrap(
            np.angle((s21_canonical - center)), discont=discont
        )
        phase_model = S21CenteredPhaseModel()
        phase_result = phase_model.fit(s21_centered_phase, self.f)
        self.fr = phase_result.best_values["fr"]
        self.Ql = phase_result.best_values["Ql"]
        # phase_result.plot_fit()

        self.phi = -np.arcsin(center.imag / radius)

        # in the denominator, use 1 for reflection and 2 for transmission/hanger mode
        Qc = self.Ql / (2 * radius * np.exp(-1j * self.phi))
        self.absQc = np.abs(Qc)
        self.Qi = 1 / ((1 / self.Ql) - (1 / Qc.real))

        # plt.cla()
        # plt.gca().set(xlabel="I", ylabel="Q", title="Complex S21")
        # plt.gca().set_aspect("equal", "datalim")
        # plt.scatter(s21_canonical.real, s21_canonical.imag, s=2, c="g", label="s21_canonical")
        # plt.plot([center.real], [center.imag], "o", ms=8, c="g")
        # plt.plot(s21_canonical[0].real, s21_canonical[0].imag, "o", ms=8, c="k")
        # circle = plt.Circle(
        # (center.real, center.imag), radius, ec="r", ls="--", fill=False, label="fit"
        # )
        # plt.gca().add_patch(circle)
        # plt.legend()
        # plt.show()

        # unpack best fit and residuals
        self.a, self.alpha = np.abs(self.orp), np.angle(self.orp)
        self.best_fit = rr_s21_hanger(
            self.f, self.fr, self.Ql, self.absQc, self.phi, a=1, alpha=0, tau=0
        )
        self.residuals = self.s21raw() - self.best_fit

        # final lmfit to calculate uncertainties
        final_model = S21Model()
        guesses = {
            "fr": {"value": self.fr},
            "Ql": {"value": self.Ql},
            "absQc": {"value": self.absQc},
            "phi": {"value": self.phi},
            "a": {"value": 1, "vary": False},
            "alpha": {"value": 0, "vary": False},
            "tau": {"value": 0, "vary": False},
        }
        final_guesses = final_model.make_params(**guesses)
        final_result = final_model.fit(
            self.s21canonical(), self.f, params=final_guesses
        )
        final_fr = final_result.uvars["fr"]
        final_absQc = final_result.uvars["absQc"]
        final_Ql = final_result.uvars["Ql"]
        final_phi = final_result.uvars["phi"]

        final_Qi = 1 / ((1 / final_Ql) - (cos(final_phi) / final_absQc))
        self.fr, self.fr_err = final_fr.n, final_fr.s
        self.Qi, self.Qi_err = final_Qi.n, final_Qi.s
        self.absQc, self.absQc_err = final_absQc.n, final_absQc.s
        self.Ql, self.Ql_err = final_Ql.n, final_Ql.s
        self.phi, self.phi_err = final_phi.n, final_phi.s

        self.best_fit = final_result.best_fit

        self.params = [
            f"fr = {self.fr:.2e}",
            f"Qi = {self.Qi:.2e}",
            f"|Qc| = {self.absQc:.2e}",
            f"Ql = {self.Ql:.2e}",
            f"phi = {self.phi:.2f}",
        ]

        if plot:
            self.plot()

    def plot(self):
        """ """
        s21 = self.s21canonical()
        plt.gca().set(xlabel="I", ylabel="Q")
        plt.gca().set_aspect("equal", "datalim")
        plt.scatter(s21.real, s21.imag, s=3, c="k", label="data")
        plt.scatter(s21.real[0], s21.imag[0], s=8, c="g")
        plt.plot(self.best_fit.real, self.best_fit.imag, ls="--", c="r", label="fit")
        plt.title(
            f"Complex S21 Trace #{self.id} [P: {self.power} dBm, T: {self.temperature} mK]"
        )
        plt.figtext(
            0.50,
            0.01,
            f"{self.params}",
            horizontalalignment="center",
            wrap=True,
            fontsize=10,
        )
        plt.legend()
        plt.tight_layout()
        plt.show()
