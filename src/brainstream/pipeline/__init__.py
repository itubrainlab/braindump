"""brAIn stream's recording pipeline, organised around g.Pype: the device
sources (:mod:`.sources`), the BIDS writers (:mod:`.writers`) and the one
session pipeline that carries them (:mod:`.session`), and the nodes the
device pages draw from (:mod:`.views`)."""
from .session import DeviceError, SessionPipeline
from .sources import AcquisitionStamps, ClockMap, PushSource
from .views import SweepTap, ViewFilter
from .writers import (BdfWriter, BidsWriter, EventLog, PhysioWriter,
                      wait_until)

__all__ = ["SessionPipeline", "DeviceError", "PushSource", "ClockMap",
           "AcquisitionStamps", "BidsWriter", "BdfWriter", "PhysioWriter",
           "EventLog", "wait_until", "SweepTap", "ViewFilter"]
