""" Script to find cable delay 'tau' and off-resonant point to calibrate raw complex resonator data (with high resolution data measured at good SNR) before extracting quality factors """

# %% IMPORTS
from pathlib import Path

import h5py
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

from circlefit import fit_circle
from delayfit import fit_delay_circular, fit_delay_linear
from resonator import fit_magnitude, fit_phase, remove_background, remove_slope

# %% DATA LOAD DETAILS
dataset_name = "W3_08_7.06_-120"
date = "20221121"
datafilename = "150601_vnasweep_7.06GHz_-70.0pow_150reps"
datafilepath = Path.cwd() / f"data/wheel/{date}/{datafilename}.hdf5"

# %% READ DATA FROM DATAFILE
with h5py.File(datafilepath, "r") as file:
    fs = file["data"]["frequency"][...]
    s21real = np.average(file["data"]["s21_imag"][...], axis=0)
    s21imag = np.average(file["data"]["s21_real"][...], axis=0)
    reps = file.attrs["repetitions"]
    power = file.attrs["powers"][0] - file.attrs["attenuation"][0]
fstep = np.average(np.diff(fs))
points = len(fs)  # number of swept data points

# %% (5) TRANSFORM RAW DATA
s21raw = s21real + 1j * s21imag
s21mag = np.abs(s21raw)
s21phase = np.unwrap(np.angle(s21raw))

# %% (6) PREPARE PLOT
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
s21_ax = fig.add_subplot(gs[:4, 3:])
s21_ax.set(ylabel="Im(S21)", title="Raw data in complex plane")
s21_ax.set_aspect("equal", "datalim")
s21_ax.locator_params(axis="both", nbins=6)
s21_ax.grid(visible=True, alpha=0.5)

# cable delay fit residuals
cdfit_res_ax = fig.add_subplot(gs[4:6, 3:])
cdfit_res_ax.set(title="Cable delay fit residuals")

# calibrated s21 in complex plane
cal_s21_ax = fig.add_subplot(gs[6:10, 3:])
cal_s21_ax.set(xlabel="Re(S21)", ylabel="Im(S21)", title="Resonance circle fit")
cal_s21_ax.set_aspect("equal", "datalim")
cal_s21_ax.locator_params(axis="both", nbins=6)
cal_s21_ax.grid(visible=True, alpha=0.5)

# circle fit residuals
circlefit_res_ax = fig.add_subplot(gs[10:12, 3:])
circlefit_res_ax.set(title="Circle fit residuals")

# %% (7) PLOT RAW DATA
raw_phase_ax.scatter(fs, s21phase, s=2, c="k", label="raw data")
raw_mag_ax.scatter(fs, s21mag, s=2, c="k", label="raw data")
s21_ax.scatter(s21real, s21imag, s=2, c="k", label="raw data")
s21_ax.plot([s21real[0]], [s21imag[0]], "o", ms=8, c="k")
fig  # to show figure inline in the interactive window

# %% (8) LINEAR CABLE DELAY FIT TO GET TAU GUESS
xpts = (points // 8, points // 8)  # for fitting linear background slope
ldfit_data, ldfit_params = fit_delay_linear(fs, s21phase, xpts, ax=raw_phase_ax)
tau_ldf, theta_ldf = ldfit_params["tau"], ldfit_params["theta"]
print(tau_ldf)
fig

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

# %% (11) CIRCULAR CABLE DELAY FIT
nfwhm_cd = 8  # number of FWHM points (out of all points) for circular cable delay fit
hchop_cd = int((nfwhm_cd * fwhm_mf) / (2 * fstep))
li_cd = max(0, fri_mf - hchop_cd)
ri_cd = min(len(fs) - 1, fri_mf + hchop_cd)
fs_cd, s21raw_cd = fs[li_cd:ri_cd], s21raw[li_cd:ri_cd]

s21nd, cdfit_params, cdfit_res = fit_delay_circular(fs_cd, s21raw_cd, tau_ldf, s21_ax)
radius_cd, center_cd = cdfit_params["radius"], cdfit_params["center"]
tau = cdfit_params["tau"]

raw_phase_ax.scatter(fs_cd, s21phase[li_cd:ri_cd], s=2, c="b", label="cdfit_data")
raw_mag_ax.scatter(fs_cd, s21mag[li_cd:ri_cd], s=2, c="b", label="cdfit_data")
s21_ax.scatter(s21raw_cd.real, s21raw_cd.imag, s=2, c="b", label="cdfit_data")
cdfit_res_ax.scatter(fs_cd, cdfit_res, s=2, c="k")
fig

# %% (12) PHASE FIT
nfwhm_pf = 3  # number of FWHM points (out of points with no cable delay) for phase fit
hchop_ph = int((nfwhm_pf * fwhm_mf) / (2 * fstep))
fri_cd = np.abs(fs_cd - fr_mf).argmin()
li_ph = max(0, fri_cd - hchop_ph)
ri_ph = min(len(fs_cd) - 1, fri_cd + hchop_ph)
fs_pf, s21nd_pf = fs_cd[li_ph:ri_ph], s21nd[li_ph:ri_ph]

s21phaseo = np.unwrap(np.angle((s21nd_pf - center_cd)))
Ql_g = fr_mf / fwhm_mf
pf_pts = 2 * hchop_ph // nfwhm_pf  # number of extreme points to guess y-offset
pfit_data, pfit_params, pfit_res = fit_phase(fs_pf, s21phaseo, pf_pts, Ql_g, pfit_ax)
fr, theta_pf = pfit_params["fr"], pfit_params["theta"]

pfit_ax.scatter(fs_pf, s21phaseo, s=2, c="g", label="data")
s21_ax.scatter(s21nd_pf.real, s21nd_pf.imag, s=2, c="g", label="pfit_data")
pfit_res_ax.scatter(fs_pf, pfit_res, s=2, c="k")
fig

# %% (13) FIND OFF-RESONANT POINT AND CALIBRATE S21
s21c, rp, orp = remove_background(s21nd_pf, theta_pf, center_cd, radius_cd, ax=s21_ax)
fig

# %% (14) SHOW CALIBRATED S21
radius, center = fit_circle(s21c)
ideal_orp = 1.0 + 1j * 0.0
rp = center + (center - ideal_orp)
circlefit_residuals = (radius - np.abs(s21c - center)) ** 2
cx, cy = center.real, center.imag

circlefit_res_ax.scatter(fs_pf, circlefit_residuals, s=2, c="k")
cal_s21_ax.scatter(s21c.real, s21c.imag, s=2, c="g", label="cal data")
cal_s21_ax.plot([s21c.real[0]], [s21c.imag[0]], "o", ms=6, c="g")
cal_s21_ax.plot([ideal_orp.real], [ideal_orp.imag], "o", ms=6, c="k")
cal_s21_ax.plot([rp.real], [rp.imag], "o", ms=6, c="r")
cal_s21_ax.plot([cx], [cy], "o", ms=6, c="g")
circle = plt.Circle((cx, cy), radius, ec="r", ls="--", fill=False, label="fit")
cal_s21_ax.add_patch(circle)
fig

# %% (15) ADD TEXT TO FIGURE AND ACTIVATE LEGEND FOR ALL AXES
raw_phase_ax.legend(loc="lower right")
raw_mag_ax.legend(loc="lower right")
s21_ax.legend()
pfit_ax.legend(loc="lower right")
cal_s21_ax.legend()

fig.text(0.6, 0.05, f"{tau = :.7e}\n{orp = :.7}")

fig

# %% (16) SAVE CALIBRATION RESULTS
savefilename = f"{dataset_name}_calibration"
savefolder = Path.cwd() / "results"
savefilepath = savefolder / savefilename
with open(str(savefilepath) + ".txt", "w+") as f:

    f.write("RESONATOR RESPONSE CALIBRATION \n")
    f.write("\n")
    f.write(f"Datafile: {date}/{datafilepath.name}\n")
    f.write(f"Repetitions: {reps}\n")
    f.write(f"Number of sweep points: {points}\n")
    f.write(f"Input power: {power:.2f} dBm\n")
    f.write("\n")

    f.write("Summary of results\n")
    f.write("\n")

    f.write(f"Resonance frequency: {fr:.3e}\n")
    f.write(f"Cable delay: {tau:.7e}\n")
    f.write(f"Off-resonant point: {orp:.7}\n")
    f.write(f"Phase fit data selection: [{li_cd + li_ph}:{li_cd + ri_ph}]\n")
    f.write("\n")

    f.write("Detailed results\n")
    f.write("\n")

    f.write("Linear phase fit to estimate cable delay\n")
    f.write(f"Tau: {tau_ldf:.3e}\n")
    f.write(f"Theta: {ldfit_params['theta']:.3}\n")
    f.write("\n")

    f.write("Lorentzian fit to get FWHM to select data for subsequent fitting\n")
    f.write(f"Full width half maxmimum: {fwhm_mf:.3e}\n")
    f.write(f"Resonance frequency: {fr_mf:.3e}\n")
    f.write(f"Peak height: {mfit_params['height']:.3}\n")
    f.write(f"Y offset: {mfit_params['ofs']:.3}\n")
    f.write(f"Impedance mismatch angle: {mfit_params['phi']:.3}\n")
    f.write("\n")

    f.write("Circle fit to remove cable delay\n")
    f.write(f"Number of FWHM selected for fitting: {nfwhm_cd}\n")
    f.write(f"Number of points selected for fitting: {len(fs_cd)}\n")
    f.write(f"Tau: {cdfit_params['tau']:.3e}\n")
    f.write(f"Circle center: {center_cd:.3}\n")
    f.write(f"Circle radius: {radius_cd:.3}\n")
    f.write("\n")

    f.write("Phase fit of S21 around origin to extract Ql\n")
    f.write(f"Number of FWHM selected for fitting: {nfwhm_pf}\n")
    f.write(f"Number of points selected for fitting: {len(fs_pf)}\n")
    f.write(f"Phase Y offset: {theta_pf:.3}\n")
    f.write("\n")

    f.write("Remove background and calibrate S21\n")
    f.write(f"Resonant point after background removal: {rp:.3}\n")
    f.write(f"Center after background removal: {center:.3}\n")
    f.write(f"Radius after background removal: {radius:.3}\n")
    f.write("\n")

savefilename
# %%
