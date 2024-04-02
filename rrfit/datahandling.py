""" """

from pathlib import Path
import pickle


class Trace:
    """Encapsulates a data trace and its acquisition and analysis record"""


class Chip:
    """ """


class Device:
    """ """

    def __init__(self, name):
        """ """
        self.name = name
        self.traces = []


class Reader:
    """
    read all data files in a given folder
    for each file, identify the file type
    for each file type, there's a load function to load data into the
    chip, resonator, and trace hierarchy
    """

    def __init__(self, loc):
        """
        loc: location to a folder containing data files
        """
        self.loc = loc
        self.read()

    def read(self):
        """ """
        for filepath in Path(self.loc).iterdir():
            if filepath.suffix == ".pickle":
                self.load_pickle(filepath)

    def load_pickle(self, filepath):
        """ """
        with open(filepath, "rb") as file:
            content = pickle.load(file)

        devices = content["inputDict"]["names"]
        for device_name in devices:
            device = Device(device_name)


class Saver:
    """ """


class Plotter:
    """ """
