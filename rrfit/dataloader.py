""" """

from pathlib import Path
import pickle
from scipy.io import loadmat
import numpy as np

from rrfit.device import Chip
from rrfit.trace import Trace


class Loader:
    """
    load all data files in a given folder
    for each file, identify the file type
    for each file type, there's a load function to load data into the
    chip, device, and trace hierarchy
    """

    def __init__(self, data_folder, calib_folder=None, devices=None):
        """
        data_folder: location to a folder containing data files
        calib_folder: location to a folder containing files for calibrating cable delay
        devices: (optional) list of device names whose data is to be loaded
        """
        # the data files in each folder belong to the same chip
        self.chip = Chip()
        self.data_folder = data_folder
        self.calib_folder = calib_folder
        self.devices = devices
        self.load()

    def load(self):
        """ """
        # do calibration, if needed
        if self.calib_folder is not None:
            self.load_calibration()

        self.load_data()

    def load_calibration(self):
        """ """
        self._load(self.calib_folder)
        # do calibration
        for device in self.chip.devices.values():
            # for each device, only the last added trace is taken to be the calibration trace
            device.calib_trace = device.traces.pop()
            # device.calib_trace.remove_cable_delay()
            # reset the device traces for data traces to be loaded in
            device.traces = []

    def load_data(self):
        """ """
        # self._load(self.data_folder)
        self.load_mat()

        # update calibration, if present
        # fit s21 only if calibration is present
        # if calibration not present,  user has to supply cable delay value and call fit_s21() manually
        # for device in self.chip.devices.values():
        #    if device.calib_trace is not None:
        #        for trace in device.traces:
        #            trace.tau = device.calib_trace.tau
        #            trace.fit_s21()

    def _load(self, folder):
        """ """
        for filepath in Path(folder).iterdir():
            content = self.load_pickle(filepath)

            if content:
                devices = content["inputDict"]["names"]
                # devices = content["acquisition_params"]["names"]  # list of loaded device names
                device_filter = self.devices if self.devices is not None else devices
                self.build_chip(content, devices, device_filter, filepath)

    def load_pickle(self, filepath: Path):
        """ """
        if filepath.suffix == ".pickle":
            with open(filepath, "rb") as file:
                return pickle.load(file)
        return {}

    def build_chip(self, content: dict, all_devices, device_filter, filepath):
        """ """
        for idx, name in enumerate(all_devices):
            if name in device_filter:
                newname = name  # filepath.stem[:4] # TEMP
                self.chip.add_device(newname)  # in case this device is new
                device = self.chip.devices[newname]
                trace = self.create_trace(content, idx)
                device.add_trace(trace)

    def create_trace(self, content: dict, idx: int):
        """idx is the device index"""
        # for RFSOC, need to convert from MHz to Hz
        f = content["data"]["f"][idx] * 1e6
        # choose re = I, im = Q convention
        re = content["data"]["IArray"][idx]
        im = content["data"]["QArray"][idx]
        power = content["inputDict"]["power"]
        # power = content["acquisition_params"]["powers"][idx]
        temperature = content["data"]["temperature_mK"]
        trace = Trace(f, re, im, power=power, temperature=temperature)
        return trace

    def load_mat(self):

        def create_trace(content, idx):
            f = content["transfreq"][0]
            power = content["power_temp"][0][0]
            temperature = content["fridgetemp_temp"][0][0]
            magnitude = 10 ** (content["transamp_temp"][0] / 20)
            phase = np.deg2rad(content["transphase_temp"][0])
            s21 = magnitude * np.exp(1j * phase)
            trace = Trace(
                idx, f, s21.real, s21.imag, power=power, temperature=temperature
            )
            return trace

        self.chip = Chip()
        for name in self.devices:
            self.chip.add_device(name)
            device = self.chip.devices[name]
            p = Path(f"{self.data_folder}/{name}")
            # for j, folder in enumerate(p.iterdir()):
            for idx, file in enumerate(p.iterdir()):
                content = loadmat(file)
                trace = create_trace(content, idx)
                device.add_trace(trace)
