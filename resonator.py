""" """

import lmfit
import numpy as np


def rrfn(fs, nports, fr, Ql, Qc, phi=0, a=1, alpha=0, tau=0):
    """
    complex resonator model
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


def gaussian_noise(signal, noisex):
    """
    add gaussian noise to signal
    signal: array of values noise is to be added to
    noisex: factor to scale gaussian noise relative to half the range of data
    """
    noise = np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    return signal + (noisex * 0.5 * (max(abs(signal)) - min(abs(signal))) * noise)


def rrmagfn(fs, ofs, height, phi, fr, fwhm):
    """
    Lorentzian fit fn for resonator magnitude response
    fs: array of probe frequencies (independent variables)
    ofs: y offset
    height: Lorentzian peak/dip to floor distance
    phi: phase factor to account for peak asymmetry
    fr: resonant frequency (peak/dip location)
    fwhm: full width half maxmimum of peak/dip
    """
    return np.abs(ofs + height * np.exp(1j * phi) / (1 + 2j * ((fs - fr) / fwhm)))


def rrphasefn(fs, fr, Ql, theta):
    """
    Arctan fit for resonator phase response around complex plane origin
    fs: array of probe frequencies (independent variables)
    fr: resonant frequency
    Ql: loaded (total) quality factor
    theta: arbitrary ohase y-offset
    note that np.arctan return real values in the interval [-pi/2, pi/2]
    """
    return theta + 2 * np.arctan(2 * Ql * (1 - fs / fr))


def remove_slope(fs, data, npoints):
    """ """
    left, right = npoints
    s21ml, fsl = data[:left], fs[:left]  # leftmost data points
    sl = (s21ml[-1] - s21ml[0]) / (fsl[-1] - fsl[0])
    yl = np.average(s21ml - sl * fsl)
    resultl = lmfit.models.LinearModel().fit(s21ml, x=fsl, slope=sl, intercept=yl)

    s21mr, fsr = data[-right:], fs[-right:]  # rightmost data points
    sr = (s21mr[-1] - s21mr[0]) / (fsr[-1] - fsr[0])
    yr = np.average(s21mr - sr * fsr)
    resultr = lmfit.models.LinearModel().fit(s21mr, x=fsr, slope=sr, intercept=yr)

    slope = (resultl.best_values["slope"] + resultr.best_values["slope"]) / 2
    yofs = (resultl.best_values["intercept"] + resultr.best_values["intercept"]) / 2
    print(f"{slope = :.3e}, {yofs = :.3e}")
    data -= slope * fs
    return data


def fit_magnitude(fs, s21mag, npoints, ax=None):
    """
    Fit resonator magnitude data to Lorentzian lineshape
    resonance feature must roughly be at the centre of the input data
    fs: array of probe frequencies (independent variables)
    s21mag: magnitude (linear) of resonator signal to be fitted
    npoints: tuple(left, right) number of left and right extreme points for yofs guess
    ax: optional matplotlib axis to plot fit on
    fs must be sorted and have same linear step size
    return array of best fit values and dict of best fit parameters
    """
    left, right = npoints

    ofs_guess = np.average((s21mag[:left] + s21mag[-right:]) / 2)
    height_guess = np.abs(np.max(s21mag) - np.min(s21mag))
    phi_guess = 4 * np.arcsin((np.max(s21mag) - ofs_guess) / height_guess)
    fr_i = (s21mag - np.abs(ofs_guess + height_guess * np.exp(1j * phi_guess))).argmin()
    fr_guess = fs[fr_i]
    #is_inverted = np.abs(s21mag.argmin() - fr_i) < np.abs(s21mag.argmax() - fr_i)
    is_inverted = np.abs(s21mag[0] - s21mag.max()) < np.abs(s21mag[-1] - s21mag.min())
    height_guess = -height_guess if is_inverted else height_guess

    hamp = height_guess / 2 + ofs_guess
    l, r = s21mag[:fr_i], s21mag[fr_i:]
    fwhm_guess = fs[fr_i + np.abs(r - hamp).argmin()] - fs[np.abs(l - hamp).argmin()]

    guesses = {
        "ofs": ofs_guess,
        "height": height_guess,
        "phi": phi_guess,
        "fr": fr_guess,
        "fwhm": fwhm_guess,
    }
    result = lmfit.Model(rrmagfn).fit(s21mag, fs=fs, **guesses)
    best_fit, best_values = result.best_fit, result.best_values

    ofs, height, phi = best_values["ofs"], best_values["height"], best_values["phi"]
    fr, fwhm = best_values["fr"], best_values["fwhm"]
    if ax is not None:
        ax.plot(fs, best_fit, ls="--", c="r", label="fit")
        ax.text(
            0.05,
            0.5,
            f"{fr = :.2e}\n{fwhm = :.2e}\n{height = :.2}\n{ofs = :.2}\n{phi = : .2}\n",
            fontsize=8,
            transform=ax.transAxes,
        )

    print(lmfit.fit_report(result))

    return best_fit, best_values, result.residual


def fit_phase(fs, s21phase, npoints, Ql_guess=None, fr_guess=None, ax=None):
    """
    Fit resonator phase data to arctan lineshape
    fs: array of probe frequencies (independent variables)
    s21phase: unwrapped phase of the complex s21 signal translated to origin
    npoints: number of extreme points used for guessing y-offset
    Ql_guess: initial guess for loaded quality factor
    ax: optional matplotlib axis to plot fit on
    fs must be sorted and have same linear step size
    return array of best fit values and dict of best fit parameters
    """
    is_shape_s = s21phase[0] < s21phase[-1]
    # determine reasonable initial guesses for fit parameters
    # fr guess is the location of max gradient of s21phase
    if fr_guess is None:
        fr_guess = fs[np.argmax(np.gradient(s21phase))]
    if Ql_guess is None:
        # Ql guess is the geometric mean of the max and min possible Ql
        Ql_guess = (fr_guess / (max(fs) - min(fs))) * np.sqrt(len(fs))
    Ql_guess = -Ql_guess if is_shape_s else Ql_guess
    # theta guess is half the the mean of npoints first and last values
    theta_guess = np.average((s21phase[:npoints] + s21phase[-npoints:]) / 2)

    guesses = {"fr": fr_guess, "Ql": Ql_guess, "theta": theta_guess}
    result = lmfit.Model(rrphasefn).fit(s21phase, fs=fs, **guesses)
    best_fit, best_values = result.best_fit, result.best_values
    fr, Ql, theta = best_values["fr"], best_values["Ql"], best_values["theta"]
    Ql = -Ql if is_shape_s else Ql
    best_values["Ql"] = Ql

    if ax is not None:
        ax.plot(fs, best_fit, ls="--", c="r", label="fit")
        ax.text(
            0.05,
            0.05 if not is_shape_s else 0.75,
            f"{fr = :.2e}\n{Ql = :.2e}\n{theta = :.2}",
            fontsize=8,
            transform=ax.transAxes,
        )

    print(result.fit_report())

    return best_fit, best_values, result.residual


def remove_background(s21data, theta, center, radius, ax=None):
    """
    Remove background from s21 data by locating the off-resonant point in complex plane
    s21data: complex resonator response with cable delay removed
    theta: background phase offset
    center: center of resonance circle in complex plane
    radius: radius of resonance circle in complex plane
    ax: optional matplotlib axis to plot points on
    return tuple of s21data at canonical position with background removed, resonant point location, and off-resonant point location
    """
    # bring theta to [-pi, pi]
    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    rp = center + radius * np.cos(theta) + 1j * radius * np.sin(theta)
    orp = center + (center - rp)
    s21canonical = s21data / orp

    if ax is not None:
        ax.plot([rp.real], [rp.imag], "o", ms=6, c="r")
        ax.plot([orp.real], [orp.imag], "o", ms=6, c="k")
        ax.text(
            0.025,
            0.25,
            f"rp = ({rp.real:.2}, {rp.imag:.2})\norp = ({orp.real:.2}, {orp.imag:.2})",
            fontsize=8,
            transform=ax.transAxes,
        )

    return s21canonical, rp, orp
