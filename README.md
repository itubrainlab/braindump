# brAIn dump

Multi-device recording for the ITU brAIn lab, into a **BIDS** dataset, on
Windows, macOS and Linux:

- **g.tec Unicorn Hybrid Black**: EEG, through [g.Pype](https://gpype.gtec.at)
  and [Rhino Recorder](https://github.com/itubrainlab/rhinorecorder)'s nodes;
- **g.tec g.HIamp**: high-density EEG (up to 256 channels, 256–4800 Hz),
  through g.Pype's GHIamp source and g.tec's gtec-gds driver;
- **Gazepoint GP3**: eye tracking, through
  [OpenGP3](https://github.com/itubrainlab/opengp3), on this computer or on
  another one over the Open Gaze API;
- **Polar H10**: ECG, heart rate and RR intervals;
- **Shimmer3 GSR+**: skin conductance and pulse (PPG);
- **LSL streams**: markers from any Lab Streaming Layer stream, as events;
- **Webcam**: any camera on the computer, as video;
- **OBS (NDI)**: OBS's output over NDI (DistroAV plugin), low latency, from
  this or another computer, as video.

Video is stored as BIDS behavioural video, following the BEP047 proposal
(`beh/…_recording-<label>_video.mp4` and its sidecar, listed in
`.bidsignore` until BEP047 is released), with every frame's time in a
`…_recording-<label>_physio.tsv.gz` beside it.

## Download

Get the package for your system from the
[latest release](https://github.com/itubrainlab/braindump/releases/latest):

| System | Package | Install |
|---|---|---|
| Windows 10 or 11, 64-bit | `brAIndump-<version>-windows-x86_64.zip` | Unpack it and double-click `install.cmd`; open brAIn dump from the Start menu. |
| macOS 14 or later, Apple Silicon | `brAIndump-<version>-macos-arm64.dmg` | Drag **brAIn dump** to Applications. The first time, right-click it and choose **Open** (the app is not notarized). |
| Linux x86_64, glibc 2.34+ (Ubuntu 22.04, Debian 12, Fedora 35 or later) | `brAIndump-<version>-linux-x86_64.tar.gz` | Unpack it and run `./install.sh`; open brAIn dump from the applications menu or run `braindump`. |

None needs administrator rights. **The first launch needs an internet
connection:** the packages contain their own Python and brAIn dump, but not
g.Pype or g.tec's Unicorn driver, which g.tec's licences do not allow to be
bundled. The first launch downloads them and the other libraries from PyPI,
at the versions the release was tested with.

## Running from source

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test,shimmer]"
.venv/bin/braindump              # choose or create a project
.venv/bin/braindump --simulate   # no hardware
.venv/bin/python -m pytest       # BIDS_VALIDATOR=.venv/bin/bids-validator-deno to validate too
```

`shimmer` installs pyshimmer (GPL-3.0), which the Shimmer3 GSR+ needs; the
rest of brAIn dump does without it.

- **Projects** (`<name>.braindump`): the devices, each one's configuration
  (saved as you change it), the BIDS output folder, the default task and the
  authors. The app opens on a project picker.
- **UI**: Rhino Recorder's toolbar and viewer, plus a left column listing the
  project's devices and the supported ones (add with +, remove with ×).
- **GP3**: an eye view (camera image locally, pupil positions remotely),
  distance feedback, calibration (5/9/13/16/25 points, full-screen targets
  here or on the tracker computer), validation (local), and a plot of the
  calibration quality.
- **Polar H10** (Bluetooth LE, via bleak): the raw ECG at 130 Hz, optionally
  the accelerometer, and the heart rate with every RR interval. BIDS physio
  `recording-ecg` (`cardiac`, µV) and `recording-accel`, with each beat in
  `recording-ecg_physioevents`.
- **LSL streams**: any Lab Streaming Layer streams, picked by name, type or
  source id (or from a list of the streams on the network), found whenever
  they appear. Every sample they send while recording becomes an event
  (`trial_type` = stream name, `value` = sample) in the run's `_events.tsv`,
  merged with the EEG's triggers and timed from the EEG's first sample.
- **Record** asks for the participant, session, task and run, then writes
  `eeg/*_eeg.bdf` (+ json/channels/events) and the eye tracking as BIDS
  eye-tracking physio under `beh/` (`recording-eye1|eye2|gaze`, plus
  physioevents and the run's events with `StimulusPresentation`), with each
  file's start time in `scans.tsv`.

## Synchronisation

Every device is time-stamped on one monotonic clock of the recording
computer. The EEG's first-sample time and its real sampling rate are
measured from every sample's acquisition time (`RecordingTiming` in
`_eeg.json`), and event onsets and `StartTime`s are given in the EEG's own
time through that rate, so they stay on the right EEG sample even as the
headset's clock drifts. A test sends a UDP trigger (placed by g.Pype) and
an LSL marker (placed by brAIn dump) at the same instant and checks they
land on the same EEG sample.

## Validating a setup

[`validation/`](validation/README.md) has a 10-minute procedure, in
PsychoPy with LSL markers, and an analysis that checks a recording: EEG
alpha with the eyes closed vs open, gaze accuracy, the N170 to faces, and
the synchronisation of EEG, markers, eyes and the screen (OBS).

## Devices by system

| | Windows | macOS | Linux |
|---|---|---|---|
| Unicorn | pair it first | found over Bluetooth | found over Bluetooth |
| g.HIamp | USB, gtec-gds driver | USB, gtec-gds driver | USB, gtec-gds driver |
| GP3, this computer | WinUSB driver via [Zadig](https://zadig.akeo.ie) once | nothing | a USB rule, installed once (asks for the password) |
| GP3, remote | nothing | nothing | nothing |
| Polar H10 | Bluetooth LE | Bluetooth LE | Bluetooth LE (BlueZ) |
| Shimmer3 GSR+ | pair it; its outgoing COM port | pair it; `/dev/cu.Shimmer3-…` | pair it and `rfcomm bind`; `/dev/rfcomm0` |
| LSL | nothing | nothing | nothing |
| Webcam | nothing | camera access, asked once | nothing (V4L2) |
| OBS over NDI | DistroAV plugin in OBS | DistroAV plugin in OBS | DistroAV plugin in OBS |

g.tec's Unicorn driver has builds for Windows x64, Linux x86_64 and Apple
Silicon Macs only.

## Packages and releases

`packaging/` builds the three packages (each needs uv, git and the
project's `.venv`; the Linux and Windows ones build on macOS or Linux):

```bash
packaging/macos/build_app.sh          # dist/brAIndump-<version>-macos-arm64.dmg
packaging/linux/build_linux.sh        # dist/brAIndump-<version>-linux-x86_64.tar.gz
packaging/windows/build_windows.sh    # dist/brAIndump-<version>-windows-x86_64.zip
```

The [Test and build](.github/workflows/build.yml) workflow runs the tests on
all three systems, builds the packages, installs each on its own system and
runs [packaging/smoke_test.py](packaging/smoke_test.py) with every device
simulated. To release: write `docs/releases/v<version>.md`, set
`__version__` in `src/braindump/__init__.py`, commit, and push a
`v<version>` tag.

## Licence

brAIn dump is MIT licensed; see [LICENSE](LICENSE). It runs on g.Pype, under
the **g.tec Non-Commercial License**: free for academic research and teaching
with a g.tec amplifier; read it before use. The Shimmer's driver, pyshimmer,
is GPL-3.0 and optional.
