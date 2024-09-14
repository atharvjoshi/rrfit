""" """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from uncertainties.umath import cos as ucos

from rrfit.circlefit import fit_background, fit_circle
from rrfit.models import S21Model, S21CenteredPhaseModel

# TODO plot residuals, separate fit and plot, show s21raw, s21nodelay, s21canonical
# TODO find a way for caller to provide experiment specific figtext


def fit_s21(s21, f, plot=False, figsize=(12, 12), **params):
    """no cable delay correction"""
    s21raw = np.copy(s21)

    # do optional cable delay correction
    tau = params.get("tau", 0)
    s21 *= np.exp(-1j * 2 * np.pi * f * tau)
    s21nodelay = np.copy(s21)

    # find off-resonant point
    discont = params.get("discont", 1.5 * np.pi)
    orp = fit_background(s21, f, discont=discont)
    s21 /= orp
    s21canonical = np.copy(s21)

    ## circle and phase fit to extract resonator parameters
    radius, center = fit_circle(s21)

    phase_model = S21CenteredPhaseModel()
    centered_phase = phase_model.center_phase(s21 - center, discont=discont)
    phase_result = phase_model.fit(centered_phase, f)
    fr_guess = phase_result.best_values["fr"]
    Ql_guess = phase_result.best_values["Ql"]

    phi_guess = -np.arcsin(center.imag / radius)

    # Qc expression valid for transmission/hanger mode
    # for reflection mode, use 1 in the denominator instead of 2
    Qc_guess = Ql_guess / (2 * radius * np.exp(-1j * phi_guess))
    absQc_guess = np.abs(Qc_guess)

    # final lmfit to find final parameter values and fitting uncertainties
    # assume that the background has been fitted out above
    s21_model = S21Model()
    guesses = {
        "fr": {"value": fr_guess},
        "Ql": {"value": Ql_guess},
        "absQc": {"value": absQc_guess},
        "phi": {"value": phi_guess},
        "a": {"value": 1, "vary": False},
        "alpha": {"value": 0, "vary": False},
        "tau": {"value": 0, "vary": False},
    }
    guesses = s21_model.make_params(**guesses)
    s21_result = s21_model.fit(s21, f, params=guesses)

    fit_failed = False
    # extract fit parameters as ufloats
    try:
        fr = s21_result.uvars["fr"]
        absQc = s21_result.uvars["absQc"]
        Ql = s21_result.uvars["Ql"]
        phi = s21_result.uvars["phi"]
        Qi = 1 / ((1 / Ql) - (ucos(phi) / absQc))
    except AttributeError as err:
        print(f"Fit failed. Details: {err}")
        fit_failed = True

    # extract fit parameters and fitting errors
    if not fit_failed:
        fit_params = {
            "fr": fr.n,
            "fr_err": fr.s,
            "Qi": Qi.n,
            "Qi_err": Qi.s,
            "absQc": absQc.n,
            "absQc_err": absQc.s,
            "Ql": Ql.n,
            "Ql_err": Ql.s,
            "phi": phi.n,
            "phi_err": phi.s,
            "background_amp": abs(orp),
            "background_phase": np.angle(orp),
        }
        fig_title = f"S21 fit [fr = {fr.n:.2g}, Qi = {Qi.n:.2g},"
        fig_title += f" Ql = {Ql.n:.2g}, |Qc| = {absQc.n:.2g}, phi = {phi.n:.2g}]"
    else:
        fit_params = {
            "fr": None,
            "fr_err": None,
            "Qi": None,
            "Qi_err": None,
            "absQc": None,
            "absQc_err": None,
            "Ql": None,
            "Ql_err": None,
            "phi": None,
            "phi_err": None,
            "background_amp": abs(orp),
            "background_phase": np.angle(orp),
        }
        fig_title = "S21 data, fit errored out"

    if plot:
        fig = plt.figure(tight_layout=True, figsize=figsize)
        fig.suptitle(fig_title)
        gs = GridSpec(6, 6, figure=fig)

        s21_ax = fig.add_subplot(gs[:3, :3])
        s21_ax.set(xlabel="Re(S21)", ylabel="Im(S21)")
        s21_ax.set(title="Circle fit")
        s21_ax.set_aspect("equal", "datalim")
        s21_ax.locator_params(axis="both", nbins=6)
        s21_ax.grid(visible=True, alpha=0.5)
        s21_ax.scatter(s21.real, s21.imag, s=3, c="k", label="data")
        s21_ax.scatter(s21.real[0], s21.imag[0], s=8, c="g")

        s21_cp_ax = fig.add_subplot(gs[:3, 3:])
        s21_cp_ax.set(xlabel="Frequency (Hz)", ylabel="arg(S21c) (rad)")
        s21_cp_ax.set(title="Centered phase fit")
        s21_cp_ax.locator_params(axis="both", nbins=6)
        s21_cp_ax.scatter(f, centered_phase, s=3, c="k", label="data")

        s21_mag_ax = fig.add_subplot(gs[3:, :3])
        s21_mag_ax.set(xlabel="Frequency (Hz)", ylabel="|S21| (dB)")
        s21_mag_ax.set(title="Fitted S21 Magnitude")
        s21_mag_ax.locator_params(axis="both", nbins=6)
        s21_mag = 20 * np.log10(np.abs(s21))
        s21_mag_ax.scatter(f, s21_mag, s=3, c="k", label="data")

        s21_phase_ax = fig.add_subplot(gs[3:, 3:])
        s21_phase_ax.set(xlabel="Frequency (Hz)", ylabel="arg(S21) (rad)")
        s21_phase_ax.set(title="Fitted S21 Phase")
        s21_phase_ax.locator_params(axis="both", nbins=6)
        s21_phase = np.unwrap(np.angle(s21))
        s21_phase_ax.scatter(f, s21_phase, s=3, c="k", label="data")

        if not fit_failed:
            s21_best_fit = s21_result.best_fit
            s21_ax.plot(
                s21_best_fit.real, s21_best_fit.imag, ls="--", c="r", label="fit"
            )
            cp_best_fit = phase_result.best_fit
            s21_cp_ax.plot(f, cp_best_fit, ls="--", c="r", label="fit")
            s21_mag_fit = 20 * np.log10(np.abs(s21_best_fit))
            s21_mag_ax.plot(f, s21_mag_fit, ls="--", c="r", label="fit")
            s21_phase_fit = np.unwrap(np.angle(s21_best_fit))
            s21_phase_ax.plot(f, s21_phase_fit, ls="--", c="r", label="fit")

        s21_ax.legend()
        s21_cp_ax.legend()
        s21_mag_ax.legend()
        s21_phase_ax.legend()
        plt.show()

    return fit_params
