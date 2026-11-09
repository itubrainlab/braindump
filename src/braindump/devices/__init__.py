"""The devices brAIn dump records, by type."""
from __future__ import annotations

from pathlib import Path

from .base import Device, DeviceConfig
from .gp3 import GP3Device
from .hiamp import HIampDevice
from .lsl import LslDevice
from .polar import PolarDevice
from .shimmer import ShimmerDevice
from .unicorn import UnicornDevice
from .video import ObsDevice, WebcamDevice

#: Supported devices, in the order the device column lists them.
DEVICE_TYPES: dict[str, type[Device]] = {
    UnicornDevice.TYPE: UnicornDevice,
    HIampDevice.TYPE: HIampDevice,
    GP3Device.TYPE: GP3Device,
    PolarDevice.TYPE: PolarDevice,
    ShimmerDevice.TYPE: ShimmerDevice,
    LslDevice.TYPE: LslDevice,
    WebcamDevice.TYPE: WebcamDevice,
    ObsDevice.TYPE: ObsDevice,
}


def make_device(device_type: str, config: dict, project_folder: Path,
                simulate: bool = False) -> Device:
    cls = DEVICE_TYPES[device_type]
    return cls(cls.CONFIG.from_dict(config), project_folder, simulate)
