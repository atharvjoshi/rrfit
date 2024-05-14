""" """

from rrfit.trace import Trace

class Chip:
    """ """

    def __init__(self):
        """ """
        self.devices: dict[str, Device] = {}  # key = device name, value = Device object

        # chip level parameters
        self.tc = None  # superconducting critical temperature

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
        self.spr = None
        self.pitch = None
        self.traces: list[Trace] = []
        self.calib_trace: Trace = None  # trace used for calibrating cable delay

    def add_trace(self, trace: Trace):
        """ """
        self.traces.append(trace)

    def __repr__(self):
        """ """
        return f"{self.__class__.__name__} '{self.name}'"
