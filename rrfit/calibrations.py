""" """

import numpy as np
import matplotlib.pyplot as plt

from rrfit.models import S21PhaseLinearModel, S21CenteredPhaseModel
from rrfit.circlefit import fit_circle


def fit_cable_delay(s21_phase, f, exclude=None):
    """
    exclude tuple(int, int): select data to exclude from cable delay fit
    """

    model = S21PhaseLinearModel()

    # fit all data to a linear model
    if exclude is None:
        result = model.fit(s21_phase, f)
        #result.plot(datafmt=".", show_init=True)
        return result.best_values["tau"]

    # fit left-most and right-most data points each to a linear model
    # return the mean tau from these two fits

    l_idx, r_idx = exclude

    # linear fit to left-most data points
    lphase, lf = s21_phase[:l_idx], f[:l_idx]
    lresult = model.fit(lphase, lf)

    # linear fit to right-most data points
    rphase, rf = s21_phase[r_idx:], f[r_idx:]
    rresult = model.fit(rphase, rf)

    ltau, rtau = lresult.best_values["tau"], rresult.best_values["tau"]

    # plt.scatter(f, s21_phase, s=2, c="k", label="data")
    # plt.plot(lf, lresult.best_fit, c="r", label="left fit")
    # plt.plot(rf, rresult.best_fit, c="r", label="right fit")
    # plt.legend()

    return (ltau + rtau) / 2


def fit_background(s21, f):
    """ """
    radius, center = fit_circle(s21)
    s21cphase = np.unwrap(np.angle((s21 - center)))

    model = S21CenteredPhaseModel()
    result = model.fit(s21cphase, f)

    theta = result.best_values["theta"]
    beta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    rp = center + radius * np.cos(beta) + 1j * radius * np.sin(beta)
    orp = center + (center - rp)

    # result.plot(
    #    datafmt=".",
    #    show_init=True,
    #    xlabel="Frequency (MHz)",
    #    ylabel="arg(S21) (rad)",
    #    data_kws={"ms": 3, "c": "k"},
    #    fit_kws={"lw": 1.5, "c": "r"},
    #    init_kws={"lw": 1.5, "c": "g"},
    # )

    # plt.cla()
    # plt.gca().set(xlabel="I", ylabel="Q", title="Complex S21")
    # plt.gca().set_aspect("equal", "datalim")
    # plt.scatter(s21.real, s21.imag, s=2, c="g", label="data")
    # plt.plot([orp.real], [orp.imag], "o", ms=8, c="k")
    # plt.plot([rp.real], [rp.imag], "o", ms=8, c="r")
    # plt.plot([center.real], [center.imag], "o", ms=8, c="g")
    # circle = plt.Circle(
    #    (center.real, center.imag), radius, ec="r", ls="--", fill=False, label="fit"
    # )
    # plt.gca().add_patch(circle)
    # plt.legend()

    return orp
