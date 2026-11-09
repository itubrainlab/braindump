"""The devices brAIn dump records, by type."""
from __future__ import annotations

from pathlib import Path

from .base import Device, DeviceConfig
from .gp3 import GP3Device
from .lsl import LslDevice
from .polar import PolarDevice
from .unicorn import UnicornDevice

#: Supported devices, in the order the device column lists them.
DEVICE_TYPES: dict[str, type[Device]] = {
    UnicornDevice.TYPE: UnicornDevice,
    GP3Device.TYPE: GP3Device,
    PolarDevice.TYPE: PolarDevice,
    LslDevice.TYPE: LslDevice,
}


def make_device(device_type: str, config: dict, project_folder: Path,
                simulate: bool = False) -> Device:
    cls = DEVICE_TYPES[device_type]
    return cls(cls.CONFIG.from_dict(config), project_folder, simulate)
