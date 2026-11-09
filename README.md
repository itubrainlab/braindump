# brAIn dump

Multi-device recording for the ITU brAIn lab. Records the **g.tec Unicorn
Hybrid Black** (EEG, through [g.Pype](https://gpype.gtec.at) and
[Rhino Recorder](https://github.com/itubrainlab/rhinorecorder)'s nodes) and the
**Gazepoint GP3** (eye tracking, through
[OpenGP3](https://github.com/itubrainlab/opengp3), either on this computer or
remotely over the Open Gaze API) into a **BIDS** dataset. It runs on Windows,
macOS and Linux.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/braindump              # choose or create a project
.venv/bin/braindump --simulate   # no hardware
.venv/bin/python -m pytest
```

- **Projects** (`<name>.braindump`): the devices, each one's configuration
  (saved as you change it), the BIDS output folder, the default task and the
  authors. The app opens on a project picker.
- **UI**: Rhino Recorder's toolbar and viewer, plus a left column listing the
  project's devices and the supported ones (add with +, remove with ×).
- **GP3**: an eye view (camera image locally, pupil positions remotely),
  distance feedback, calibration (5/9/13/16/25 points, full-screen targets
  here or on the tracker computer), validation (local), and a plot of the
  calibration quality.
- **LSL streams**: any Lab Streaming Layer streams, picked by name, type or
  source id (or from a list of the streams on the network), found whenever
  they appear. Every sample they send while recording becomes an event
  (`trial_type` = stream name, `value` = sample) in the run's `_events.tsv`,
  merged with the EEG's triggers and timed from the EEG's first sample.
- **Record** asks for the participant, session, task and run, then writes
  `eeg/*_eeg.bdf` (+ json/channels/events) and the eye tracking as BIDS 1.10
  physio (`recording-eye1|eye2|gaze`, plus physioevents), with each file's
  start time in `scans.tsv`.
