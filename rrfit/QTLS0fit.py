""" functions to fit QTLS0 from Qint vs nbar or FFS vs temp data"""

import numpy as np
import lmfit
from scipy.constants import k, hbar
import matplotlib.pyplot as plt


def getPhotonNumber(freq0, QInt, avgQC, msmtP, zFeed=50, zResonator=50):
    # Edited 10/24/2022 by KDC - added feedline and resonator impedance to account for case where resonator impedance is not equal to feedline impedance.
    pre_B = 2 / (hbar * ((2 * np.pi * freq0) ** 2))
    impedanceFactor = zFeed / zResonator
    Q = 1 / (1 / QInt + 1 / avgQC)

    photonN = pre_B * impedanceFactor * ((Q**2) / avgQC) * msmtP

    return photonN


def QIntVsNbar(params, freq0, QInt, avgQC, msmtP, zFeed, zResonator, temp0):
    # Added 10/24/2022 by KDC, same idea as original function but different approach (uses getPhotonNumber).
    QTLS0 = params["QTLS0"].value
    nc = params["nc"].value
    QOther = params["QOther"].value
    beta = params["beta"].value

    omega = 2 * np.pi * freq0
    tanhTerm = np.tanh(hbar * omega / (2 * k * temp0))
    photonN = getPhotonNumber(
        freq0=freq0,
        QInt=QInt,
        avgQC=avgQC,
        msmtP=msmtP,
        zFeed=zFeed,
        zResonator=zResonator,
    )

    QTLS = QTLS0 / tanhTerm * np.sqrt(1 + np.power(photonN / nc, beta))
    QIntFromPower = 1 / (1 / QTLS + 1 / QOther)
    return QIntFromPower


def QIntVsNbar_error_function(
    params, freq0, QInt, QIntErr, avgQC, msmtP, zFeed, zResonator, temp0
):
    # Added 10/24/2022 by KDC, same idea as original function but different approach (uses getPhotonNumber).
    QIntFromPower = QIntVsNbar(
        params, freq0, QInt, avgQC, msmtP, zFeed, zResonator, temp0
    )
    return (QInt - QIntFromPower) / QIntErr


def QIntFromNbar(params, freq0Avg, temp0Avg, nBar):
    QTLS0 = params["QTLS0"].value
    nc = params["nc"].value
    QOther = params["QOther"].value
    beta = params["beta"].value

    omega = 2 * np.pi * freq0Avg
    tanhTerm = np.tanh(hbar * omega / (2 * k * temp0Avg))
    QTLS = QTLS0 / tanhTerm * np.sqrt(1 + np.power(nBar / nc, beta))
    QInt = 1 / (1 / QTLS + 1 / QOther)
    return QInt


def fit_qlts0(device):
    """ """
    # TODO change hardcoded stuff, especially init guesses

    trs = [tr for tr in device.traces if not tr.is_excluded]
    msmtPArray = np.asarray(
        [np.power(10, (tr.power - device.attenuation - 30) / 10) for tr in trs]
    )  # power into fridge
    freq0Array = np.asarray([tr.fr for tr in trs])
    freq0Avg = np.mean(freq0Array)
    temp0Array = np.asarray([tr.temperature for tr in trs])
    temp0Avg = np.mean(temp0Array)
    QIntArray = np.asarray([tr.Qi for tr in trs])  # Qint
    QIntErrArray = np.asarray([tr.Qi_err for tr in trs])  # std error of Qint
    QCArray = np.asarray([tr.absQc for tr in trs])
    QCErrArray = np.asarray([tr.absQc_err for tr in trs])
    QtotArray = np.asarray([tr.Ql for tr in trs])
    zFeedArray = np.asarray([50 for tr in trs])
    zResonator = 50
    zResonatorArray = np.asarray([zResonator for tr in trs])
    avgQC = np.mean(QCArray)
    avgQCArray = np.asarray([avgQC for tr in trs])

    nbarArray = []

    for tr in trs:
        freq0 = tr.fr
        QInt = tr.Qi
        msmtP = np.power(10, (tr.power - device.attenuation - 30) / 10)
        nbar = getPhotonNumber(
            freq0=freq0,
            QInt=QInt,
            avgQC=avgQC,
            msmtP=msmtP,
            zFeed=50,
            zResonator=zResonator,
        )
        nbarArray.append(nbar)

    nbarArray = np.asarray(nbarArray)
    initParams = lmfit.Parameters()
    initParams.add("QTLS0", value=5e6, min=0, vary=True)
    initParams.add("nc", value=1, min=0.01, vary=True)
    initParams.add("QOther", value=5e7, min=0, vary=True)
    initParams.add("beta", value=1, min=0, vary=True)
    args = [
        freq0Array,
        QIntArray,
        QIntErrArray,
        avgQCArray,
        msmtPArray,
        zFeedArray,
        zResonatorArray,
        temp0Array,
    ]
    results = lmfit.minimize(QIntVsNbar_error_function, initParams, args=args)
    #         results = hta.least_squares(hta.QIntVsNbar_error_function, initParams, args=args, max_nfev=1e6)
    nBars = np.logspace(np.log10(min(nbarArray)), np.log10(max(nbarArray)), 1000)
    params = results.params
    print(str(device.name) + ":")
    params.pretty_print()
    fig, ax = plt.subplots(figsize=(8, 5))
    #         fig, ax = plt.subplots(dpi=500)
    ax.errorbar(nbarArray, QIntArray, yerr=QIntErrArray, fmt="o")

    ax.plot(nBars, QIntFromNbar(params, freq0Avg, temp0Avg, nBars))
    ax.set_ylabel("Q_Int", fontsize=20)
    ax.set_yscale("log")
    ax.set_xlabel("n", fontsize=20)
    ax.set_xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_title(f"{device.name} ", fontsize=20)
    # fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    # ax.plot(nbarArray, QtotArray, 'o')
    # ax.set_ylabel('Q_tot', fontsize=20)
    # ax.set_yscale('log')
    # ax.set_xlabel('n', fontsize=20)
    # ax.set_xscale('log')
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # ax.set_title(str(dev.name) + ', f = {0:0.2f} GHz'.format(freq0Avg/1e9))
