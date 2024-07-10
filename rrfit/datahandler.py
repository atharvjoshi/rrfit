""" """

# TODO save calib files in a different folder
# TODO ensure ALL traces have device name attribute in hdf5 files

from dataclasses import dataclass
from pathlib import Path
import h5py


@dataclass
class Trace:
    id: int = None
    device_name: str = None
    frequency: list[float] = None
    s21real: list[float] = None
    s21imag: list[float] = None
    power: float = None
    temperature: float = None
    temperature_err: float = None
    background_amp: float = None
    background_phase: float = None
    tau: float = None
    fr: float = None
    fr_err: float = None
    Qi: float = None
    Qi_err: float = None
    Ql: float = None
    Ql_err: float = None
    absQc: float = None
    absQc_err: float = None
    phi: float = None
    phi_err: float = None
    is_excluded: bool = None


@dataclass
class Device:
    name: str = None
    pitch: float = None
    traces: list[Trace] = None
    line_attenuation: float = None


def load_data(*folders: Path, **devices: Device):
    """ """
    for folder in folders:
        for path in Path(folder).iterdir():
            # ignore non-hdf5 files
            if not path.suffix in (".h5", ".hdf5", ".hdf"):
                continue
            trace = load_trace(path)
            device = devices[trace.device_name]
            device.traces.append(trace)

    for device in devices.values():
        print(f"Found {len(device.traces)} traces for device '{device.name}'")
        for idx, trace in enumerate(device.traces):
            trace.id = idx


def load_trace(path: Path) -> Trace:
    """ """
    with h5py.File(path) as file:
        return Trace(
            device_name=file.attrs["device_name"],
            frequency=file["frequency"][:],
            s21real=file["s21real"][:],
            s21imag=file["s21imag"][:],
            power=file.attrs.get("input_power"),
            temperature=file.attrs.get("temp_avg"),
            temperature_err=file.attrs.get("temp_std"),
            background_amp=file.attrs.get("background_amp"),
            background_phase=file.attrs.get("background_phase"),
            tau=file.attrs.get("tau"),
            fr=file.attrs.get("fr"),
            fr_err=file.attrs.get("fr_err"),
            Qi=file.attrs.get("Qi"),
            Qi_err=file.attrs.get("Qi_err"),
            Ql=file.attrs.get("Ql"),
            Ql_err=file.attrs.get("Ql_err"),
            absQc=file.attrs.get("absQc"),
            absQc_err=file.attrs.get("absQc_err"),
            phi=file.attrs.get("phi"),
            phi_err=file.attrs.get("phi_err"),
        )
