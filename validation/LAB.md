# Validating the lab's setup

This is the [validation procedure](README.md) for the lab's booth: two
computers, a 64-channel g.HIamp, a Gazepoint GP3 HD, OBS and a camera. One
session with any participant (about 15 minutes) checks five things:

1. **EEG alpha:** eyes closed against eyes open (band power, first).
2. **EEG signal quality:** every channel's noise.
3. **Eye-tracker accuracy:** at the fixation crosses.
4. **N170:** to faces.
5. **Synchronisation:** of the EEG, the eye tracker, the screen recording,
   the camera and the LSL markers, with each other.

PsychoPy runs the whole procedure, including the eye tracker's calibration.
The analysis then gives each check PASS or FAIL.

## The setup

```
 STIMULUS COMPUTER                          RECORDING COMPUTER
 ┌──────────────────────────┐   network    ┌──────────────────────────┐
 │ PsychoPy  (procedure,    │── LSL ──────▶│ brAIn stream               │
 │           LSL markers,   │              │  ├ g.HIamp (USB)         │
 │           GP3 calibration)│              │  │   64 ch, 64 = HEOG    │
 │ opengp3 server ◀ GP3 HD  │── Open Gaze ▶│  ├ GP3 (remote)          │
 │           (USB)          │    API       │  ├ LSL markers           │
 │ OBS + DistroAV (NDI)     │── NDI ──────▶│  ├ Screen Capture (OBS)  │
 │ the participant's screen │              │  └ Video Stream (camera) │
 └──────────────────────────┘              └──────────────────────────┘
```

| Device | Where | Settings |
|---|---|---|
| g.HIamp | recording computer, USB | 64 channels at 512 Hz. Channels 1–63 EEG, channel 64 bipolar (horizontal EOG). Amplifier filters 0.01–100 Hz band-pass and 50 Hz notch |
| GP3 HD | stimulus computer, USB, under `opengp3 server` | read by brAIn stream over the network (Open Gaze API, port 4242). PsychoPy runs its calibration on the same server |
| OBS | stimulus computer: Display Capture of the participant's screen, NDI output (DistroAV) | received by brAIn stream's Screen Capture over the network |
| Camera | recording computer, USB, pointed at the participant | brAIn stream's Video Stream, recording as `recording-camera` |
| PsychoPy | stimulus computer | sends the markers over LSL (stream `brAInstreamValidation`) |

Both computers must be on the same wired network, with LSL, NDI and port
4242 allowed through their firewalls. Use wired Ethernet: Wi-Fi makes NDI's
and the eye tracker's delays uneven.

## Once: setting up

### Recording computer

1. **brAIn stream from source**, with the analysis's figures:

   ```bash
   pip install -e ".[validation]"
   ```

2. **g.tec's driver.** brAIn stream installs gtec-gds. On Windows, also
   install g.tec's g.HIamp driver (g.NEEDaccess).
3. **The cap's channel names.** Write them in a text file, in the order of
   the amplifier's inputs: one per line, 64 lines, the last one `HEOG`. For
   example:

   ```
   Fp1
   Fp2
   …
   PO7
   PO8
   …
   HEOG
   ```

   The analysis looks for PO7 and PO8 (N170) and for O1, Oz, O2, POz and Pz
   (alpha) by name. Without names, it falls back to every EEG channel and
   the checks get weaker.
4. **The project.** With the g.HIamp switched on and connected:

   ```bash
   python validation/make_project.py "~/brAIn stream/lab-validation" --lab \
       --stimulus-host <stimulus computer's address> \
       --electrodes cap64.txt --screen-width-cm <screen width>
   ```

   `--lab` sets up:
   - **g.HIamp:** 64 channels at 512 Hz; channel 64 recorded as EOG
     (`--channel-type 64=EOG`); the amplifier's 0.01–100 Hz band-pass and
     50 Hz notch, chosen from those the amplifier offers. If it doesn't
     offer them, the script lists the ones it does.
   - **GP3:** remote, at the stimulus computer.
   - **LSL streams:** listening for `brAInstreamValidation`.
   - **Screen Capture:** OBS over NDI.
   - **Video Stream:** the first camera. Give `--webcam "<name>"` to choose
     another.

   Give the participant's screen's physical width: accuracy is measured in
   degrees. Add `--reference`/`--ground` to describe them in the recording.
   To check the settings, open the project in brAIn stream: the g.HIamp's
   page, Settings, shows the channels and the amplifier's filters.

### Stimulus computer

1. **OpenGP3** for the GP3 HD:

   ```bash
   pip install "opengp3 @ git+https://github.com/itubrainlab/opengp3@v0.5.0"
   ```

   (On Windows, install UsbDk once; see OpenGP3's README.)
2. **[PsychoPy](https://www.psychopy.org/download.html)**, standalone. It
   includes pylsl. Copy `validation/face_validation.py` onto this computer,
   with a folder of **at least 30 face photographs**. The lab's set is in
   the uncanny-face-questionnaire study; this repository doesn't include
   them.
3. **OBS** with the [DistroAV](https://github.com/DistroAV/DistroAV)
   plugin:
   - capture the participant's screen (Display Capture);
   - in **Settings → Video**, give the canvas the screen's shape (e.g.
     1920×1080), and fit the capture to it (Edit → Transform → Fit to
     screen);
   - turn on **Tools → NDI Output Settings → Main Output**.

### The camera

It films the participant. Its timing is measured on the participant's
eyes: the analysis finds them in its video (MediaPipe's face landmarks, as
OpenGP3's webcam eye tracking does) and times each jump to a new cross,
against the same jump in the eye tracker's gaze. For that, the face should
fill a good part of the picture, lit from the front, with the eyes
visible (no glare on glasses).

The screen's flashes are a second measure, when their light on the
participant shows in the video. That happens in a dim room; in a bright
one they are simply not counted.

### The EOG (channel 64)

A bipolar pair at the outer corners of the eyes (horizontal EOG), wired to
channel 64. The analysis compares it with the eye tracker's gaze, sample by
sample, with no marker in between. This is the most direct check that the
EEG and the eye tracker are on one clock. If the lab wires channel 64
vertically (above and below an eye), the analysis follows the vertical
gaze instead.

## A session

On the **stimulus computer**:

1. Start the eye tracker's server, on the participant's screen (display
   0 here):

   ```bash
   opengp3 server --display 0
   ```

2. Start OBS, and check its NDI output is on.

On the **recording computer**:

3. Start brAIn stream and open the lab-validation project.
4. Fit the cap and the EOG electrodes. Connect the g.HIamp: every channel
   in the data viewer should show clean EEG, without drift or 50 Hz.
5. **Connect all** (the plug button).
   - **GP3 page:** the eye view shows both eyes and the distance bar is
     green. Seat the participant about 65 cm from the screen.
   - **Screen Capture:** shows the participant's screen.
   - **Video Stream:** shows the participant (and the screen).

On the **stimulus computer**:

6. Start the procedure with PsychoPy's Python, from a terminal (or open it in
   PsychoPy's Coder and Run):

   ```bash
   python face_validation.py --faces <faces folder> --participant 01 --screen 0
   ```

   It waits until brAIn stream receives its markers.

On the **recording computer**:

7. Press **Record**, with participant `01` and task `facevalidation`.

On the **stimulus computer**:

8. Press space, and give the participant the keyboard. The procedure runs:

   | Step | |
   |---|---|
   | Flashes | 10 white flashes of the whole screen |
   | Eyes open | 30 s, looking at a dot |
   | Eyes closed | 30 s, until a tone |
   | Calibration | 9 points, drawn by PsychoPy and run by the GP3 HD's server |
   | Moving targets | 2 × 10 s, a dot to follow with the eyes |
   | Faces | 30: a cross at a random place (2 s), the face there (1 s), a blank (1 s) |
   | Moving targets | 2 × 10 s again |
   | Flashes | 10 again |

9. After the debriefing screen, stop recording in brAIn stream.

## Checking the recording

On the recording computer:

```bash
python validation/analyse.py "~/brAIn stream/lab-validation/bids" --sub 01
```

It prints each check and writes `report.md`, with figures, in
`bids/derivatives/validation/sub-01/`. It exits non-zero if any check fails.

| # | Check | Passes when |
|---|---|---|
| 1 | Alpha | posterior alpha (8–13 Hz) ≥ 1.5 × with the eyes closed |
| 2 | EEG signal quality | ≥ 90 % of the EEG channels, and every alpha and N170 site, under 50 µV of 1–30 Hz noise during the faces |
| 3 | Gaze accuracy | median error ≤ 1.5° at the crosses, ≥ 80 % of crosses within 1.5° |
| 4 | N170 | a negative peak at PO7/PO8 (average reference, EOG left out) at 130–200 ms, p < 0.05 |
| 5 | Synchronisation | all of the following |

What the synchronisation check compares:

| Pair | How | Passes when |
|---|---|---|
| markers in `eeg/` and `beh/` | identical | within 1 ms |
| eye tracker ↔ markers | saccades to each new cross | 80–500 ms after it |
| eye tracker ↔ markers | the gaze following the moving targets | −50 to 150 ms behind them |
| **EEG ↔ eye tracker** | the EOG (channel 64) against the gaze, cross-correlated | within 40 ms |
| EEG ↔ markers | the N170's latency | inside 130–200 ms (check 4) |
| screen recording ↔ markers | the faces and the flashes in OBS's video | −20 to 200 ms, IQR < 50 ms |
| **camera ↔ eye tracker** | the eyes' jumps to each new cross, in the camera's video and in the gaze | within 60 ms, IQR < 100 ms |
| camera ↔ markers | the flashes' light on the participant, when it shows | within 60 ms, IQR < 70 ms |
| markers | each recorded once | no duplicates |
| every recording | covers the procedure | start to end |

When the screen recording or the camera is late but steady, the report
says how many ms to add to its **Delay** in brAIn stream (Screen Capture or
Video Stream, Settings). Record again after changing it.

### When a check fails

| Failing | Look at |
|---|---|
| Alpha | The posterior electrodes (check 2); whether the participant closed the eyes. A few people have little alpha. |
| EEG signal quality | The channels listed: gel, reseat, check the reference and ground. |
| Gaze accuracy | The valid gaze by screen height in the report. If the top or the bottom is lost, change the GP3 HD's angle or the participant's height and distance, then recalibrate. Check the screen width given to the project. |
| N170 | Too few clean epochs (beyond ±100 µV are left out), PO7/PO8 named wrong, or a timing offset (see synchronisation). |
| EOG ↔ gaze | Late gaze: the eye tracker's samples are out of step with the EEG. Check the network (wired), and that `opengp3 server` is 0.5 or later. If the EOG doesn't follow the gaze at all (r below 0.3), check its electrodes and that it is channel 64. |
| Moving targets | Gaze far behind (or ahead of) the targets with the saccades fine: the same as above. |
| Flashes in the screen recording | Not found: OBS isn't capturing the participant's screen, or NDI isn't arriving. Late and steady: set the Screen Capture's Delay to the value given. |
| Camera ↔ eye tracker | Too few jumps found: the face is too small or too dark in the picture; move the camera closer or light the face. Late and steady: add the value given to the Video Stream's Delay. Uneven (large IQR): the camera buffers frames; try a lower resolution or frame rate, or another camera. |
| Markers recorded once | A marker recorded twice is a brAIn stream problem: report it. The other checks count each marker once. |

## Trying it without the hardware

On one computer, everything simulated:

```bash
python validation/make_project.py /tmp/lab-try --lab --simulate
python face_validation.py --faces <faces folder> --dry-run --auto
```

`--dry-run` runs a short version in a window: 4 s rests, 6 faces, 3 flashes
and 1 moving target each time. With simulated devices, the analysis finds
every file and checks the markers and coverage. The physiological checks
fail, because a simulator has no eyes to close or faces to see.
`tests/test_validation_analysis.py` tests the analysis itself on synthetic
lab recordings with known timing.
