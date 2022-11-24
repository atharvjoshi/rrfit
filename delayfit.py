""" """

import lmfit
import numpy as np
import matplotlib.pyplot as plt

from circlefit import fit_circle


def fit_delay_linear(fs, s21phase, npoints, ax=None):
    """
    fs: array of probe frequencies (independent variables)
    s21phase: unwrapped phase of the complex s21 signal
    npoints: tuple(left, right) num data points on left and right extremes to be fitted
    ax: optional matplotlib axis to plot fit on
    return array of best fit values and dict of best fit parameters
    """

    def linear(fs, tau, theta):
        """
        linear model which gives an initial guess for correcting tau
        fit this with the unwrapped phase of complex resonator signal
        fs: array of probe frequencies (independent variables)
        tau: frequency-dependent phase shift due to cable delay
        theta: arbitrary phase offset
        """
        return 2 * np.pi * fs * tau + theta

    left, right = npoints
    is_shape_s = s21phase[0] < s21phase[-1]
    s21pl, fsl = s21phase[:left], fs[:left]  # leftmost data points
    taul = (s21pl[-1] - s21pl[0]) / (2 * np.pi * (fsl[-1] - fsl[0]))
    thetal = np.average(s21pl - taul / (2 * np.pi) * fsl)
    resultl = lmfit.Model(linear).fit(s21pl, fs=fsl, tau=taul, theta=thetal)
    best_fit_l, best_values_l = resultl.best_fit, resultl.best_values

    s21pr, fsr = s21phase[-right:], fs[-right:]  # rightmost data points
    taur = (s21pr[-1] - s21pr[0]) / (2 * np.pi * (fsr[-1] - fsr[0]))
    thetar = np.average(s21pr - taur / (2 * np.pi) * fsr)
    resultr = lmfit.Model(linear).fit(s21pr, fs=fsr, tau=taur, theta=thetar)
    best_fit_r, best_values_r = resultr.best_fit, resultr.best_values
    tau = (best_values_l["tau"] + best_values_r["tau"]) / 2
    theta = (best_values_l["theta"] + best_values_r["theta"]) / 2

    if ax is not None:
        ax.plot(fsl, best_fit_l, ls="--", c="r", label="fit")
        ax.plot(fsr, best_fit_r, ls="--", c="r")
        ax.text(
            0.05,
            0.05 if not is_shape_s else 0.75,
            f"{tau = :.3e}\n{theta = :.2}",
            fontsize=8,
            transform=ax.transAxes,
        )

    best_fit = (best_fit_l + best_fit_r) / 2
    best_values = {"tau": tau, "theta": theta}
    return best_fit, best_values


def fit_delay_circular(fs, s21, tau, ax=None):
    """
    fs: array of probe frequencies (independent variables)
    s21: complex s21 data from which cable delay is to be removed
    tau: initial guess for frequency-dependent phase shift due to cable delay
    ax: optional matplotlib axis to plot fit on
    return array of s21 values corrected for cable delay and final tau value
    """

    def circular(ps, fs, s21):
        """ """
        tau = ps["tau"]
        circle = s21 * np.exp(2j * np.pi * fs * -tau)
        radius, center = fit_circle(circle)
        return (radius - np.abs(circle - center)) ** 2

    params = lmfit.Parameters()
    params.add("tau", value=tau, min=0.5*tau, max=1.5*tau)

    result = lmfit.minimize(circular, params, args=(fs, s21))
    tau = result.params.valuesdict()["tau"]
    s21nd = s21 * np.exp(2j * np.pi * fs * -tau)
    radius, center = fit_circle(s21nd)
    cx, cy = center.real, center.imag
    params = {"center": center, "radius": radius, "tau": tau}
    residuals = (radius - np.abs(s21nd - center)) ** 2 

    if ax is not None:
        ax.text(
            0.025,
            0.75,
            f"{radius = :.2}\n{tau = :.2e}\ncenter = ({cx:.2}, {cy:.2})",
            fontsize=8,
            transform=ax.transAxes,
        )
        circle = plt.Circle((cx, cy), radius, ec="r", ls="--", fill=False)
        ax.add_patch(circle)
        ax.plot([cx], [cy], "o", ms=6, c="g")
        ax.scatter(s21nd.real, s21nd.imag, s=2, c="m", label="no cable delay")
        ax.plot([s21nd.real[0]], [s21nd.imag[0]], "o", ms=6, c="m")

    print(lmfit.fit_report(result))

    return s21nd, params, residuals

if __name__ == "__main__":
    """ """

    