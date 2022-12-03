""" Standalone code cells to run the fitting procedure for extracting quality factors from reflection and transmission measurements of microwave resonators, after cable delay and background have been calibrated. """

# %% (1) IMPORTS
from pathlib import Path

import h5py
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

from circlefit import fit_circle
from resonator import fit_magnitude, fit_phase, remove_background, remove_slope

# %% (2) SET MEASUREMENT MODE - REFLECTION OR TRANSMISSION
nports = 1  # 1 for reflection, 2 for transmission, other values not allowed
if nports == 1:
    meas_mode = "Reflection"
elif nports == 2:
    meas_mode = "Transmission"
else:
    raise ValueError("nports value must be 1 (reflection) or 2 (transmission)")

# %% (3) SET CALIBRATION PARAMETERS
# cable delay (s)
tau = 0

# off-resonant point
orp = 0.07109979+0.1113638j

# [left:right] to select data for phase fit
pfit_slice = slice(85, 311)
start, stop = pfit_slice.start, pfit_slice.stop

# %% (4) DATA LOAD DETAILS
dataset_name = "W3_10_6.87_-170"
date = "20221124"
datafilename = "160006_vnasweep_6.87GHz_-1.7e+02pow_1500reps"
datafilepath = Path.cwd() / f"data/wheel/{date}/{datafilename}.hdf5"

# %% (5) READ DATA FROM DATAFILE
with h5py.File(datafilepath, "r") as file:
    fs = file["data"]["frequency"][...]
    s21real = np.average(file["data"]["s21_imag"][...], axis=0)
    s21imag = np.average(file["data"]["s21_real"][...], axis=0)
    reps = file.attrs["repetitions"]
    power = file.attrs["powers"][0] - file.attrs["attenuation"][0]
fstep = np.average(np.diff(fs))
points = len(fs)  # number of swept data points

# %% (6) TRANSFORM RAW DATA
s21raw = s21real + 1j * s21imag
s21raw_pf, fs_pf = s21raw[pfit_slice], fs[pfit_slice]
s21cal = (s21raw_pf * np.exp(2j * np.pi * fs_pf * -tau)) / orp
s21mag = np.abs(s21raw)
s21phase = np.unwrap(np.angle(s21raw))

# %% (7) PREPARE PLOT
fig = plt.figure(tight_layout=True, figsize=(16, 16))
fig.suptitle(f"{dataset_name} [{fstep:.2e} step, {points = }, {power}dBm, {reps = }]")
gs = GridSpec(13, 6, figure=fig)

# raw phase plot
raw_phase_ax = fig.add_subplot(gs[:3, :3])
raw_phase_ax.set(ylabel="arg(S21) (rad)", title="Raw phase")
raw_phase_ax.locator_params(axis="both", nbins=6)

# raw magnitude plot
raw_mag_ax = fig.add_subplot(gs[3:6, :3])
raw_mag_ax.set(ylabel="abs(S21)", title="Raw magnitude")
raw_mag_ax.locator_params(axis="both", nbins=6)

# magnitude fit residuals
mfit_res_ax = fig.add_subplot(gs[6:8, :3])
mfit_res_ax.set(title="Magnitude fit residuals")

# s21 around origin phase fit
pfit_ax = fig.add_subplot(gs[8:11, :3])
pfit_ax.set(xlabel="Frequency (Hz)", ylabel="arg(S21) (rad)", title="Phase fit")
pfit_ax.locator_params(axis="both", nbins=6)

# phase fit residuals
pfit_res_ax = fig.add_subplot(gs[11:, :3])
pfit_res_ax.set(title="Phase fit residuals")

# raw s21 data in complex plane
s21_ax = fig.add_subplot(gs[:5, 3:])
s21_ax.set(ylabel="Im(S21)", title="Raw data in the complex plane")
s21_ax.set_aspect("equal", "datalim")
s21_ax.locator_params(axis="both", nbins=6)
s21_ax.grid(visible=True, alpha=0.5)

# calibrated s21 in complex plane
circlefit_ax = fig.add_subplot(gs[5:10, 3:])
circlefit_ax.set(xlabel="Re(S21)", ylabel="Im(S21)", title="Resonance circle fit")
circlefit_ax.set_aspect("equal", "datalim")
circlefit_ax.locator_params(axis="both", nbins=6)
circlefit_ax.grid(visible=True, alpha=0.5)

# circle fit residuals
circlefit_res_ax = fig.add_subplot(gs[10:12, 3:])
circlefit_res_ax.set(title="Circle fit residuals")

# %% (8) PLOT RAW AND CALIBRATED DATA
raw_phase_ax.scatter(fs, s21phase, s=2, c="k", label="raw data")
raw_phase_ax.scatter(fs_pf, s21phase[pfit_slice], s=2, c="g", label="pfit data")

raw_mag_ax.scatter(fs, s21mag, s=2, c="k", label="raw data")
raw_mag_ax.scatter(fs_pf, s21mag[pfit_slice], s=2, c="g", label="pfit data")

s21_ax.plot([s21real[0]], [s21imag[0]], "o", ms=8, c="k")
s21_ax.scatter(s21real, s21imag, s=2, c="k", label="raw data")
s21_ax.scatter(s21raw_pf.real, s21raw_pf.imag, s=2, c="g", label="pfit data")

circlefit_ax.scatter(s21cal.real, s21cal.imag, s=2, c="g", label="data")
circlefit_ax.plot([s21cal.real[0]], [s21cal.imag[0]], "o", ms=8, c="g")

xpts = (points // 8, points // 8)  # for fitting linear background slope

fig  # to show figure inline in the interactive window

# %% (9) REMOVE LINEAR BACKGROUND FROM MAGNITUDE DATA
#s21mag = remove_slope(fs, s21mag, xpts)
#raw_mag_ax.clear()
#raw_mag_ax.scatter(fs, s21mag, s=2, c="m")
#fig

# %% (10) LORENTZIAN MAGNITUDE FIT TO GET FWHM
mfit_data, mfit_params, mfit_residuals = fit_magnitude(fs, s21mag, xpts, ax=raw_mag_ax)
fr_mf, fwhm_mf = mfit_params["fr"], abs(mfit_params["fwhm"])
fri_mf = np.abs(fs - fr_mf).argmin()

mfit_res_ax.scatter(fs, mfit_residuals, s=2, c="k")
fig

# %% (11) CIRCLE FIT TO FIND CENTER AND RADIUS OF CALIBRATED S21
radius, center = fit_circle(s21cal)
cx, cy = center.real, center.imag
circlefit_residuals = (radius - np.abs(s21cal - center)) ** 2

circlefit_res_ax.scatter(fs_pf, circlefit_residuals, s=2, c="k")
circlefit_ax.plot([cx], [cy], "o", ms=6, c="g")
circle = plt.Circle((cx, cy), radius, ec="r", ls="--", fill=False, label="fit")
circlefit_ax.add_patch(circle)
fig

# %% (12) FIND OFF-RESONANT POINT AFTER A PHASE FIT
s21phaseo = np.unwrap(np.angle((s21cal - center)))
Ql_g = fr_mf / fwhm_mf
pfit_data, pfit_params, pfit_res = fit_phase(fs_pf, s21phaseo, start, Ql_g, pfit_ax)
fr, Ql, theta_pf = pfit_params["fr"], pfit_params["Ql"], pfit_params["theta"]
_, rp_cf, orp_cf = remove_background(s21cal, theta_pf, center, radius, ax=circlefit_ax)

pfit_ax.scatter(fs_pf, s21phaseo, s=2, c="g", label="data")
pfit_res_ax.scatter(fs_pf, pfit_res, s=2, c="k")
fig

# %% (13) EXTRACT PHI, |Qc|, Qi
phi = -np.arcsin((orp_cf.imag - center.imag) / radius)
Qc = Ql / (nports * radius * np.exp(-1j * phi))
absQc = np.abs(Qc)
Qi = 1 / ((1 / Ql) - (1 / Qc.real))

# %% (14) ADD TEXT TO FIGURE AND ACTIVATE LEGEND FOR ALL AXES
raw_phase_ax.legend(loc="lower right")
raw_mag_ax.legend(loc="lower right")
s21_ax.legend()
pfit_ax.legend(loc="lower right")
circlefit_ax.legend()

text1 = f"{tau = :.3e}\norp_cal = {orp:.3}\n{orp_cf = :.3}\n{fr = :.3e}"
text2 = f"{Ql = :.3e}\n{Qi = :.3e}\n|Qc| = {absQc:.3e}\n{phi = :.3}"
fig.text(0.6, 0.025, text1)
fig.text(0.8, 0.025, text2)

fig

# %% (15) SAVE CALIBRATION RESULTS
savefolder = Path.cwd() / "results"
savefilepath = savefolder / dataset_name
with open(str(savefilepath) + ".txt", "w+") as f:

    f.write("RESONATOR RESPONSE FIT TO EXTRACT QUALITY FACTORS \n")
    f.write("\n")
    f.write(f"Datafile: {date}/{datafilepath.name}\n")
    f.write(f"Measurement mode: {meas_mode}\n")
    f.write(f"Repetitions: {reps}\n")
    f.write(f"Number of sweep points: {points}\n")
    f.write(f"Input power: {power:.2f} dBm\n")
    f.write("\n")

    f.write("Summary of results\n")
    f.write("\n")

    f.write(f"Resonance frequency: {fr:.3e}\n")
    f.write(f"Loaded quality factor: {Ql:.3e}\n")
    f.write(f"Internal quality factor: {Qi:.3e}\n")
    f.write(f"Absolute value of coupling quality factor: {absQc:.3e}\n")
    f.write(f"Impedance mismatch angle phi: {phi:.3}\n")

    f.write("\n")

    f.write("Calibration parameters\n")
    f.write(f"Cable delay: {tau:.7e}\n")
    f.write(f"Off-resonant point: {orp:.7}\n")
    f.write(f"Phase fit data selection: [{start}:{stop}]\n")
    f.write("\n")

    f.write("Lorentzian fit to get Ql guess\n")
    f.write(f"Full width half maxmimum: {fwhm_mf:.3e}\n")
    f.write(f"Resonance frequency: {fr_mf:.3e}\n")
    f.write(f"Peak height: {mfit_params['height']:.3}\n")
    f.write(f"Y offset: {mfit_params['ofs']:.3}\n")
    f.write(f"Impedance mismatch angle: {mfit_params['phi']:.3}\n")
    f.write("\n")

    f.write("Phase fit of S21 around origin to extract Ql\n")
    f.write(f"Phase Y offset: {theta_pf:.3}\n")
    f.write("\n")

    f.write("Circle fit\n")
    f.write(f"Resonant point: ({rp_cf.real:.3}, {rp_cf.imag:.3})\n")
    f.write(f"Off-resonant point: ({orp_cf.real:.3}, {orp_cf.imag:.3})\n")
    f.write(f"Center: {center:.3}\n")
    f.write(f"Radius: {radius:.3}\n")
    f.write(f"Complex coupling quality factor: {Qc:.3}\n")
    f.write(f"Radius after phi correction: {radius / np.abs(np.cos(phi)):.3}\n")
    f.write("\n")

dataset_name

# %%
