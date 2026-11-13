"""One recording across every running device, into the project's BIDS
dataset."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

from .bids import NA, Dataset, Entities, write_json, write_tsv
from . import clock
from .devices.base import (Device, Event, RecordedFiles, RecordingTarget,
                           Timeline)
from .project import Project


@dataclass
class Subject:
    """Who and what is recorded, as asked when Record is pressed."""

    sub: str
    ses: str = ""
    task: str = "rest"
    run: Optional[int] = None
    age: str = ""
    sex: str = ""
    handedness: str = ""
    notes: str = ""
    task_description: str = ""
    instructions: str = ""
    #: Head circumference in cm, for the EEG ("" when not measured).
    head_circumference: str = ""

    def entities(self) -> Entities:
        return Entities(sub=self.sub, task=self.task, ses=self.ses,
                        run=self.run)


@dataclass
class RecordingResult:
    entities: Entities
    started: float
    duration: float
    files: list[Path] = field(default_factory=list)
    devices: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def dataset_for(project: Project) -> Dataset:
    from .devices import DEVICE_TYPES
    devices = [f"{DEVICE_TYPES[t].NAME}: {DEVICE_TYPES[t].DESCRIPTION}"
               for t in project.devices if t in DEVICE_TYPES]
    return Dataset(project.output_path, project.name, project.authors,
                   project.description, project.license, devices)


def common_metadata(project: Project) -> dict:
    """Sidecar fields every data file gets."""
    out = {}
    for key, value in (("InstitutionName", project.institution_name),
                       ("InstitutionAddress", project.institution_address),
                       ("InstitutionalDepartmentName",
                        project.institution_department)):
        if value:
            out[key] = value
    return out


class Session:
    """Starts and stops a recording on several devices together."""

    def __init__(self, project: Project, devices: list[Device]):
        self.project = project
        self.devices = list(devices)
        self.dataset = dataset_for(project)
        self.subject: Optional[Subject] = None
        self.target: Optional[RecordingTarget] = None
        self._recording: list[Device] = []
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return bool(self._recording)

    def planned_files(self, subject: Subject) -> list[Path]:
        """Files this recording would write that already exist."""
        e = subject.entities()
        folder = self.dataset.session_dir(e)
        if not folder.is_dir():
            return []
        stem = e.stem()
        return [p for p in folder.rglob(f"{stem}_*")]

    def start(self, subject: Subject) -> list[Path]:
        """Begin recording on every running device; returns the files."""
        with self._lock:
            if self._recording:
                raise RuntimeError("Already recording.")
            devices = [d for d in self.devices if d.running]
            if not devices:
                raise RuntimeError("No device is running.")
            e = subject.entities()
            datatype = "eeg" if any(d.MODALITY == "eeg" for d in devices) \
                else "beh"
            self.dataset.ensure()
            common = common_metadata(self.project)
            if subject.task_description:
                common["TaskDescription"] = subject.task_description
            if subject.instructions:
                common["Instructions"] = subject.instructions
            eeg_meta = {}
            try:
                if subject.head_circumference:
                    eeg_meta["HeadCircumference"] = float(
                        subject.head_circumference)
            except ValueError:
                pass
            target = RecordingTarget(self.dataset, e, datatype, clock.now(),
                                     self.project.power_line_frequency,
                                     common, eeg_meta)
            files: list[Path] = []
            started: list[Device] = []
            try:
                # Two EEG amplifiers at once would write the same
                # _eeg.bdf: each gets its own acq- entity (acq-unicorn,
                # acq-hiamp), as BIDS has it for one modality recorded
                # two ways.
                several = sum(1 for d in devices if d.MODALITY == "eeg") > 1
                for device in devices:
                    own = target
                    if several and device.MODALITY == "eeg":
                        own = replace(target, entities=replace(
                            e, acq=device.TYPE))
                    files += device.begin_recording(own)
                    started.append(device)
            except BaseException:
                for device in started:
                    try:
                        device.end_recording()
                    except Exception:  # noqa: BLE001
                        pass
                raise
            self.subject, self.target = subject, target
            self._recording = started
            return files

    def stop(self) -> Optional[RecordingResult]:
        """End the recording on every device and complete the dataset."""
        with self._lock:
            devices, target, subject = (self._recording, self.target,
                                        self.subject)
            if not devices or target is None:
                return None
            self._recording = []
            ended: list[tuple[Device, RecordedFiles]] = []
            problems = []
            for device in devices:
                try:
                    ended.append((device, device.end_recording()))
                except Exception as error:  # noqa: BLE001
                    problems.append(f"{device.NAME}: {error}")
            # The recording's time base (BIDS StartTime, event onsets):
            # the EEG's own, measured over the recording, or plain seconds
            # from the earliest first sample.
            eeg = [r.timeline for d, r in ended
                   if d.MODALITY == "eeg" and r.timeline is not None]
            firsts = [r.first_sample for _, r in ended if r.first_sample]
            reference = eeg[0] if eeg else (Timeline(min(firsts))
                                             if firsts else None)
            result = RecordingResult(target.entities, target.started,
                                     clock.now() - target.started,
                                     problems=problems)
            scans = []
            for device, recorded in ended:
                try:
                    recorded.finalize(reference)
                except Exception as error:  # noqa: BLE001
                    problems.append(f"{device.NAME}: {error}")
                scans += recorded.scans
                result.files += [f for f in recorded.files if f.exists()]
                result.devices[device.NAME] = recorded.samples
            e = target.entities
            events = [ev for _, r in ended for ev in r.events]
            extra = {}
            for _, r in ended:
                extra.update(r.events_metadata)
            # The run's events go with its data: into the EEG's folder,
            # and into beh/ when something is there (eye tracking always
            # is: its StimulusPresentation belongs in that events.json).
            folders = set()
            if events or eeg or extra:
                folders.add(target.datatype)
            if any(d.MODALITY in ("eyetrack", "video") for d, _ in ended):
                folders.add("beh")
            for folder in sorted(folders):
                try:
                    result.files += self._write_events(
                        target, folder, events,
                        reference or Timeline(target.started),
                        has_eeg=bool(eeg) and folder == "eeg", extra=extra)
                except Exception as error:  # noqa: BLE001
                    problems.append(f"events.tsv: {error}")
            self.dataset.add_participant(e.sub, {
                "age": subject.age, "sex": subject.sex,
                "handedness": subject.handedness})
            moment = datetime.fromtimestamp(reference.t0 if reference
                                            else target.started)
            self.dataset.add_session(e, moment)
            self.dataset.add_scans(e, [s for s in scans if s[0].exists()])
            if subject.notes:
                # Not a BIDS file: kept under sourcedata/, which BIDS
                # leaves free.
                notes = (self.dataset.root / "sourcedata" / "notes"
                         / f"{e.stem()}_notes.txt")
                notes.parent.mkdir(parents=True, exist_ok=True)
                notes.write_text(subject.notes + "\n", encoding="utf-8")
                result.files.append(notes)
            self.subject = self.target = None
            return result


    def _write_events(self, target: RecordingTarget, datatype: str,
                      events: list[Event],
                      reference: Timeline, has_eeg: bool,
                      extra: dict) -> list[Path]:
        """The run's events.tsv: every device's events, with onsets from
        the reference device's first sample (the EEG's, when recorded)."""
        path = self.dataset.path(target.entities, datatype, "events",
                                 ".tsv")
        columns = ["onset", "duration", "trial_type", "value", "source"]
        if has_eeg:
            columns.append("sample")
        rows = []
        for ev in sorted(events, key=lambda ev: ev.time):
            onset = reference.seconds(ev.time)
            row = [round(onset, 6), round(ev.duration, 6), ev.trial_type,
                   ev.value or NA, ev.source or NA]
            if has_eeg:
                row.append(reference.sample(ev.time))
            rows.append(row)
        write_tsv(path, columns, rows)
        sidecar = path.with_suffix(".json")
        meta = {
            "onset": {"Description": "Seconds from the first sample of the "
                                     "recording, in the EEG's own time "
                                     "(sample / SamplingFrequency) when EEG "
                                     "was recorded", "Units": "s"},
            "duration": {"Description": "Duration of the event",
                         "Units": "s"},
            "trial_type": {"Description": "\"trigger\" for the EEG's "
                                          "trigger input, else the name of "
                                          "the LSL stream that sent it"},
            "value": {"Description": "The trigger value, or the LSL "
                                     "sample (channels comma-separated)"},
            "source": {"Description": "Where the event came from: "
                                      "\"trigger\" or \"lsl:<stream>\""},
        }
        if has_eeg:
            meta["sample"] = {"Description": "EEG sample of the onset "
                                             "(from 0)"}
        meta.update(extra)
        if datatype == "beh":
            # The behavioural events file is the run's only data sidecar.
            meta = {"TaskName": target.entities.task,
                    **target.common_metadata, **meta}
        write_json(sidecar, meta)
        return [path, sidecar]
