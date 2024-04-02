""" """

from pathlib import Path
import pickle


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
        self.real = real
        self.imag = imag

        # control variables
        self.power = None
        self.temperature = None

        # fit parameters
        self.cable_delay = None

        for name, value in kwargs.items():
            setattr(self, name, value)

    def s21(self, sign=1):
        """
        sign must be 1 or -1
        """
        return self.real + sign * 1j * self.imag

    # s21maglog, s21phase

    def fit_cable_delay(self):
        """ """
        # TODO

    def fit_s21(self):
        """ """
        # TODO


class Chip:
    """ """

    # TODO add chip level params etc Tc

    def __init__(self):
        """ """
        self.devices: dict[str, Device] = {}  # key = device name, value = Device object

    def add_device(self, name: str):
        """ """
        if name not in self.devices.keys():
            device = Device(name)
            self.devices[name] = device


class Device:
    """ """

    def __init__(self, name):
        """ """
        self.name = name
        self.traces: list[Trace] = []

        # the cable delay experienced by this Device due to electrical length of measurement chain
        self.cable_delay = 0

    def add_trace(self, trace: Trace):
        """ """
        self.traces.append(trace)


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
        self.data_folder = data_folder
        self.calib_folder = calib_folder
        self.devices = devices
        self.load()

    def load(self):
        """ """
        # the data files in each folder belong to the same chip
        chip = Chip()

        # do calibration, if needed
        if self.calib_folder is not None:
            self.load_calibration(chip)

        self.load_data(chip)

        return chip

    def load_calibration(self, chip: Chip):
        """ """
        self._load(self.calib_folder, chip)

        for device in chip.devices.values():
            # for each device, only the last added trace is taken to be the calibration trace
            calib_trace = device.traces.pop()
            calib_trace.fit_cable_delay()
            device.cable_delay = calib_trace.cable_delay
            # reset the device traces for data traces to be loaded in
            device.traces = []

    def load_data(self, chip: Chip):
        """ """
        self._load(self.data_folder, chip)

        for device in chip.devices.values():
            for trace in device.traces:
                trace.cable_delay = device.cable_delay
                trace.fit_s21()

    def _load(self, folder, chip):
        """ """
        for filepath in Path(folder).iterdir():
            content = self.load_pickle(filepath)

            if content:
                devices = content["inputDict"]["names"]  # list of loaded device names
                device_filter = self.devices if self.devices is not None else devices
                self.build_chip(chip, content, devices, device_filter)

    def load_pickle(self, filepath: Path):
        """ """
        if filepath.suffix == ".pickle":
            return pickle.load(filepath)
        return {}

    def build_chip(self, chip: Chip, content: dict, all_devices, device_filter):
        """ """
        for idx, name in enumerate(all_devices):
            if name in device_filter:
                chip.add_device(name)  # in case this device is new
                device = chip.devices[name]
                trace = self.create_trace(content, idx)
                device.add_trace(trace)

    def create_trace(self, content: dict, idx: int):
        """idx is the device index"""
        f = content["data"]["f"][idx]
        # choose re = I, im = Q convention
        re = content["data"]["IArray"][idx]
        im = content["data"]["QArray"][idx]
        power = content["data"]["power"][idx]
        temperature = content["data"]["temperature_mK"]
        trace = Trace(f, re, im, power=power, temperature=temperature)
        return trace
