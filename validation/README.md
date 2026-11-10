# Validating a brAIn dump setup

This procedure checks that brAIn dump records the lab's signals **correctly**
and **in step with each other**:

- **EEG**: g.tec Unicorn;
- **gaze**: Gazepoint GP3;
- **the screen**: OBS over NDI;
- **the experiment's events**: PsychoPy's LSL markers.

A participant (anyone; 10 minutes) rests with eyes open and closed, follows
the eye tracker's calibration, and looks at 30 faces. The analysis then
checks four things in the recording:

| # | Check | Passes when |
|---|---|---|
| 1 | **Alpha**: eyes closed vs open | posterior alpha (8–13 Hz) power is ≥ 1.5 × higher with the eyes closed |
| 2 | **Gaze accuracy** at the fixation crosses | median error ≤ 1.5°, and ≥ 80 % of crosses within 1.5° |
| 3 | **N170** after the faces | a negative peak at PO7/PO8 between 130 and 200 ms, below zero across trials (p < 0.05) |
| 4 | **Synchronisation** of EEG, markers, eyes and screen | see below |

Synchronisation is checked several independent ways, each against the
markers:

- **Markers:** they are identical in `eeg/` and `beh/` (to 1 ms).
- **Eyes:** they reach each new cross 80–500 ms after its marker. A saccade
  can't come before its target, and doesn't take half a second.
- **Eyes and EEG:** gaze is lost within 3 s of "eyes closed" and back within
  3 s of the tone, and the EEG's alpha rises within 3 s of "eyes closed".
- **Screen:** each face appears in the screen recording 0–200 ms after its
  marker (display, capture and NDI), with an IQR under 50 ms.
- **Coverage:** EEG, gaze and screen each cover the whole procedure.

The N170's latency is itself a check that the EEG and the markers line up:
a misalignment of more than a few tens of ms moves it out of its window.

## Files

| File | |
|---|---|
| [`face_validation.py`](face_validation.py) | The procedure, in PsychoPy. Every step is an LSL marker (stream `brAIndumpValidation`) stamped at the screen flip that shows it. |
| [`make_project.py`](make_project.py) | Makes the brAIn dump project the procedure records with. |
| [`analyse.py`](analyse.py) | Checks a recording; writes `report.md` with figures. |

## The procedure

1. **Instructions**
2. **Resting, eyes open**: 30 s, looking at a dot in the centre.
3. **Resting, eyes closed**: 30 s; a tone says when to open them.
4. **Eye-tracking calibration**: 9 points, drawn by PsychoPy and run by the
   GP3's server (OpenGP3's client-drawn calibration).
5. **30 faces** in random order, each one:
   - a fixation cross at a random place on the screen (2 s);
   - the face, centred where the cross was (1 s);
   - a blank screen (1 s).

   The face appears where the participant is already looking, so no eye
   movement disturbs the N170.
6. **Debriefing**

The markers are `procedure_start`, `instructions`, `rest_open_start` and
`rest_open_end`, `rest_closed_start` and `rest_closed_end` (the tone),
`calibration_start`, `calibration_point n= x= y=`, `calibration_result`,
`calibration_end`, then for each trial `fixation trial= x= y=`,
`face trial= image= x= y=` and `blank trial=`, then `debriefing` and
`procedure_end`. Positions are fractions of the screen from the top left,
the same frame as the GP3's gaze.

## Setting up (once)

**The computer.** Install brAIn dump from source, with the analysis's
figures:

```bash
pip install -e ".[validation]"
```

Then install [PsychoPy](https://www.psychopy.org/download.html). The
standalone app includes pylsl.

**The eye tracker.** The GP3 runs under OpenGP3's server, so that PsychoPy
can draw its calibration and brAIn dump can record it at the same time. On
the computer the GP3 is connected to, run:

```bash
opengp3 server --display 0
```

Use the `--display` the participant's screen is. Gazepoint Control also
works, in its place.

**OBS.** Install the [DistroAV](https://github.com/DistroAV/DistroAV) plugin,
capture the participant's screen (Display Capture), and turn on
**Tools → NDI Output Settings → Main Output**.

**The face images.** You need a folder of at least 30 face photographs. The
lab's questionnaire study has 36:
`uncanny-face-questionnaire/website/static/images/human`. Check their
licence before sharing them; this repository does not include any.

**The project.** Make it once:

```bash
python validation/make_project.py "~/brAIn dump/validation" \
    --unicorn UN-2024.10.18 --gp3-host 127.0.0.1 --screen-width-cm 53
```

Give the screen's physical width: the accuracy is measured in degrees. The
script sets up four devices:

- the Unicorn;
- the GP3, read over the Open Gaze API;
- the **LSL streams** device, listening for `brAIndumpValidation`;
- **OBS (NDI)**, recording as `recording-screen`.

## A session

1. **Start everything.** Start `opengp3 server` and OBS, then brAIn dump; open
   the validation project. Seat the participant about 65 cm from the screen.
2. **Fit the Unicorn.** The data viewer's quality dots should be green.
3. **Connect all** (the plug button) in brAIn dump. On the GP3 page, check
   the eye view (both eyes, the distance bar in the green).
4. **Start the procedure**, with PsychoPy's Python (or open it in PsychoPy's
   Coder and Run):

   ```bash
   python face_validation.py --faces <faces folder> --participant 01 --screen 0
   ```

   It waits until brAIn dump receives its marker stream.
5. **Press Record** in brAIn dump: participant `01`, task `facevalidation`.
   Then press space in PsychoPy.
6. **Stop.** After the debriefing, stop recording in brAIn dump.

PsychoPy also writes its own log of the markers, with their LSL times, in
`validation/data/`.

## Checking the recording

```bash
python validation/analyse.py "~/brAIn dump/validation/bids" --sub 01
```

This prints each check and writes `report.md`, with the alpha spectra, the
gaze offsets, the face ERP and the latency histograms, in
`bids/derivatives/validation/sub-01/`. It exits non-zero if any check fails.

### When a check fails

| Failing | Look at |
|---|---|
| Alpha | The electrodes (quality dots); whether the participant really closed the eyes. A few people have little alpha. |
| Gaze accuracy | Recalibrate; check the screen width and distance in the GP3 settings, and that the procedure runs on the display the GP3's server tracks. |
| N170 | Noise at PO7/PO8, or too few clean epochs (> ±100 µV are left out). A latency outside 130–200 ms points to a timing offset (see Synchronisation). |
| Markers in eeg/ and beh/ | A recording error: report it. |
| Saccades to the crosses | Negative or > 500 ms: the gaze and the markers are out of step. Check the GP3's clock mapping (remote: the network); the latency settings. |
| Faces in the screen recording | Not found: is OBS capturing the stimulus screen, in full? Late or spread out: NDI's latency (set the OBS delay setting to its median). |

## Without hardware

`--dry-run` gives a short version (4 s rests, 6 faces) in a window. `--auto`
goes on by itself without key presses. Use them to test the chain with
brAIn dump's simulated devices. The project script's `--simulate` makes every
device simulated. The analysis then finds every file and checks the markers
and coverage, while the physiological checks fail: a simulator has no eyes
to close or faces to see.

The analysis itself is tested (`tests/test_validation_analysis.py`) on
synthetic recordings with known alpha, gaze, N170 and timing. It passes a
good recording, and catches each fault:

- no alpha effect;
- inaccurate gaze;
- no N170;
- markers 300 ms out of step.

**A better timing reference.** For a measurement of the display's own
latency, put a photodiode on the screen corner and record it with the EEG.
The face markers could then be checked against the light itself.
