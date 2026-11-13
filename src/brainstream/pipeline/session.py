"""The session pipeline: one g.Pype pipeline carrying every connected
device.

Each device contributes its nodes (:meth:`Device.build
<brainstream.devices.base.Device.build>`): its g.Pype source, the taps its
page draws from, its filters and its BIDS writers. The amplifier, when
there is one, owns the pipeline's master timeline. A running g.Pype
pipeline cannot gain or lose nodes, so connecting or disconnecting a
device stops the pipeline and starts it again with the new set; and when
one device fails, the pipeline stops, and with it every device.

Recordings begin and end through g.Pype's control API: the session
invokes each BIDS writer's ``begin`` and ``end`` actions
(:mod:`brainstream.pipeline.writers`).
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable, Iterable, Optional

import gpype as gp

if TYPE_CHECKING:  # pragma: no cover
    from ..devices.base import Device


class DeviceError(RuntimeError):
    """A device could not connect, or stopped; names the device."""

    def __init__(self, device: "Device", message: str):
        super().__init__(f"{device.NAME}: {message}")
        self.device = device
        self.message = message


class SessionPipeline:
    """One g.Pype pipeline for every connected device of a project.

    :meth:`connect` and :meth:`disconnect` block (opening a device can
    take seconds), so a user interface calls them off its main thread.
    """

    def __init__(self):
        self.pipeline: Optional[gp.Pipeline] = None
        #: The devices in the running pipeline, in the order they joined.
        self.devices: list["Device"] = []
        self._lock = threading.RLock()
        #: Called with (device, message) when the pipeline stops on a
        #: device's error.
        self.on_failure: Callable[["Device", str], None] = \
            lambda device, message: None

    # -- membership --------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.pipeline is not None

    def connect(self, devices: Iterable["Device"]) -> None:
        """Add ``devices`` to the pipeline (restarting it)."""
        with self._lock:
            wanted = list(self.devices)
            for device in devices:
                if device not in wanted:
                    wanted.append(device)
            if wanted != self.devices:
                self._restart(wanted)

    def disconnect(self, devices: Iterable["Device"]) -> None:
        """Take ``devices`` out of the pipeline (restarting it with the
        others, if any)."""
        with self._lock:
            leaving = set(id(d) for d in devices)
            wanted = [d for d in self.devices if id(d) not in leaving]
            if len(wanted) != len(self.devices):
                self._restart(wanted)

    def stop(self) -> None:
        """Stop the pipeline and every device."""
        with self._lock:
            self._restart([])

    def _restart(self, wanted: list["Device"]) -> None:
        recording = [d.NAME for d in self.devices if d.recording]
        if recording:
            raise RuntimeError(
                f"Stop recording before connecting or disconnecting a "
                f"device ({', '.join(recording)} recording).")
        staying = [d for d in wanted if d in self.devices]
        self._shutdown(keep=staying)
        if wanted:
            self._start(wanted)

    # -- lifecycle ---------------------------------------------------------------

    def _start(self, devices: list["Device"]) -> None:
        from ..devices.base import IDLE, RUNNING, STARTING
        for device in devices:
            device.error = None
            device._set_state(STARTING)
        pipeline = gp.Pipeline()
        opened: list["Device"] = []
        current = None
        try:
            for device in devices:
                current = device
                device.build(pipeline)
            current = None
            pipeline.add_error_handler(self._pipeline_failed)
            pipeline.start()
            for device in devices:
                current = device
                device.opened()
                opened.append(device)
            current = None
        except BaseException as error:
            for device in reversed(opened):
                try:
                    device.closing()
                except Exception:  # noqa: BLE001
                    pass
            try:
                pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
            for device in devices:
                device.closed()
                device._set_state(IDLE)
            if current is None:
                current = self._owner(str(error), devices)
            if current is not None and not isinstance(error, DeviceError):
                raise DeviceError(current, str(error)) from error
            raise
        self.pipeline = pipeline
        self.devices = list(devices)
        for device in devices:
            device._set_state(RUNNING)

    def _shutdown(self, keep: list["Device"]) -> None:
        from ..devices.base import IDLE, STARTING, STOPPING
        pipeline, devices = self.pipeline, self.devices
        self.pipeline, self.devices = None, []
        if pipeline is None:
            return
        for device in devices:
            device._set_state(STARTING if device in keep else STOPPING)
        for device in reversed(devices):
            try:
                device.closing()
            except Exception:  # noqa: BLE001 - stopping regardless
                pass
        try:
            pipeline.stop()
        finally:
            for device in devices:
                device.closed()
                if device not in keep:
                    device._set_state(IDLE)

    # -- failures --------------------------------------------------------------------

    @staticmethod
    def _owner(text: str, devices: list["Device"]) -> Optional["Device"]:
        """The device whose node a g.Pype message names."""
        for device in devices:
            if f"{device.TYPE}_" in text:
                return device
        return devices[0] if len(devices) == 1 else None

    def _pipeline_failed(self, entry) -> None:
        try:
            message = str(entry["message"])
            node = str(entry.get("node") or entry.get("name") or "")
        except Exception:  # noqa: BLE001
            message, node = str(entry), ""
        device = self._owner(node or message, list(self.devices))
        self.fail(device, message)

    def fail(self, device: Optional["Device"], message: str) -> None:
        """Stop everything because ``device`` failed (from any thread)."""
        def work():
            with self._lock:
                if device is not None and device not in self.devices:
                    return
                if device is not None:
                    device.error = message
                try:
                    self._shutdown(keep=[])
                except Exception:  # noqa: BLE001
                    pass
            if device is not None:
                device.on_error(message)
                self.on_failure(device, message)
        threading.Thread(target=work, daemon=True,
                         name="session-failure").start()

    # -- control -------------------------------------------------------------------

    def invoke(self, node, action: str, arg: Optional[dict] = None):
        """Run a node's g.Pype action (a writer's ``begin`` or ``end``)."""
        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError("The session pipeline is not running.")
        return pipeline.invoke_action(node, action, dict(arg or {}))
