"""Asked every time Record is pressed: who is recorded, in which session."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QGroupBox, QLabel, QLineEdit,
                               QPlainTextEdit, QSpinBox, QVBoxLayout)

from ..bids import Dataset, Entities, is_label, read_tsv
from ..recording import Subject
from . import theme

_LABEL = QRegularExpressionValidator(QRegularExpression("[A-Za-z0-9]*"))


class RecordDialog(QDialog):
    """Participant, session, task and run of a recording, in BIDS
    labels. The run is the next free one."""

    #: What the last recording was asked, offered again.
    last: Optional[Subject] = None

    def __init__(self, dataset: Dataset, default_task: str,
                 devices: list[str], parent=None, tasks: dict | None = None,
                 eeg: bool = False):
        super().__init__(parent)
        self.dataset = dataset
        self.tasks = tasks or {}
        self.subject: Optional[Subject] = None
        self.setWindowTitle("Record")
        self.setMinimumWidth(480)
        column = QVBoxLayout(self)
        intro = QLabel(f"Recording from: <b>{', '.join(devices)}</b>")
        intro.setWordWrap(True)
        column.addWidget(intro)

        form = QFormLayout()
        self.sub = QLineEdit()
        self.sub.setValidator(_LABEL)
        self.sub.setPlaceholderText("e.g. 01")
        self.ses = QLineEdit()
        self.ses.setValidator(_LABEL)
        self.ses.setPlaceholderText("e.g. 1 (optional)")
        self.task = QLineEdit(default_task)
        self.task.setValidator(_LABEL)
        self.run = QSpinBox()
        self.run.setRange(1, 999)
        form.addRow("Participant", self.sub)
        form.addRow("Session", self.ses)
        form.addRow("Task", self.task)
        form.addRow("Run", self.run)
        column.addLayout(form)

        task = QGroupBox("Task (remembered per task in the project)")
        about_task = QFormLayout(task)
        self.task_description = QLineEdit()
        self.task_description.setPlaceholderText("What the participant does")
        self.instructions = QLineEdit()
        self.instructions.setPlaceholderText("What they were told")
        about_task.addRow("Description", self.task_description)
        about_task.addRow("Instructions", self.instructions)
        column.addWidget(task)

        about = QGroupBox("Participant details (participants.tsv, optional)")
        details = QFormLayout(about)
        self.age = QLineEdit()
        self.age.setValidator(QRegularExpressionValidator(
            QRegularExpression(r"\d{0,3}")))
        self.sex = QComboBox()
        self.sex.addItems(["", "F", "M", "O"])
        self.hand = QComboBox()
        self.hand.addItems(["", "R", "L", "A"])
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Notes on this recording (kept in "
                                      "sourcedata/notes)")
        self.notes.setFixedHeight(56)
        details.addRow("Age", self.age)
        details.addRow("Sex", self.sex)
        details.addRow("Handedness", self.hand)
        self.head = QLineEdit()
        self.head.setValidator(QRegularExpressionValidator(
            QRegularExpression(r"\d{0,3}(\.\d?)?")))
        self.head.setPlaceholderText("cm, for the EEG (optional)")
        if eeg:
            details.addRow("Head circumference", self.head)
        details.addRow("Notes", self.notes)
        column.addWidget(about)

        self.preview = QLabel()
        self.preview.setObjectName("Hint")
        self.preview.setWordWrap(True)
        column.addWidget(self.preview)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText("Start recording")
        ok.setObjectName("Danger")
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        column.addWidget(self.buttons)

        last = RecordDialog.last
        if last is not None:
            self.sub.setText(last.sub)
            self.ses.setText(last.ses)
            self.task.setText(last.task)
        for edit in (self.sub, self.ses, self.task):
            edit.textChanged.connect(self._labels_changed)
        self.task.textChanged.connect(self._task_changed)
        self._task_changed()
        self.sub.textChanged.connect(self._known_participant)
        self.run.valueChanged.connect(self._update)
        self._labels_changed()
        self._known_participant()
        self.sub.setFocus()

    def _labels_changed(self):
        sub, ses, task = (self.sub.text(), self.ses.text(), self.task.text())
        if is_label(sub) and is_label(task):
            self.run.setValue(self.dataset.next_run(sub, ses, task))
        self._update()

    def _task_changed(self):
        info = self.tasks.get(self.task.text(), {})
        self.task_description.setText(info.get("TaskDescription", ""))
        self.instructions.setText(info.get("Instructions", ""))

    def _known_participant(self):
        """Fill in what participants.tsv already knows."""
        _, rows = read_tsv(self.dataset.root / "participants.tsv")
        row = next((r for r in rows
                    if r.get("participant_id") == f"sub-{self.sub.text()}"),
                   None)
        if row is None:
            return
        if row.get("age", "n/a") != "n/a":
            self.age.setText(row["age"])
        for combo, key in ((self.sex, "sex"), (self.hand, "handedness")):
            value = row.get(key, "")
            if combo.findText(value) >= 0:
                combo.setCurrentText(value)

    def _entities(self) -> Optional[Entities]:
        try:
            return Entities(self.sub.text(), self.task.text(),
                            self.ses.text(), self.run.value())
        except ValueError:
            return None

    def _update(self):
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        e = self._entities()
        if e is None:
            ok.setEnabled(False)
            self.preview.setText("Participant and task are needed: letters "
                                 "and digits only.")
            return
        ok.setEnabled(True)
        folder = self.dataset.session_dir(e)
        existing = list(folder.rglob(f"{e.stem()}_*")) if folder.is_dir() \
            else []
        text = f"Into {folder}<br>as <b>{e.stem()}_…</b>"
        if existing:
            text += (f"<br><span style='color:{theme.WARNING}'>This run "
                     f"exists already ({len(existing)} files): recording "
                     f"replaces them.</span>")
        self.preview.setText(text)

    def _accept(self):
        e = self._entities()
        if e is None:
            return
        self.subject = Subject(sub=e.sub, ses=e.ses, task=e.task, run=e.run,
                               age=self.age.text(),
                               sex=self.sex.currentText(),
                               handedness=self.hand.currentText(),
                               notes=self.notes.toPlainText().strip(),
                               task_description=self.task_description.text()
                               .strip(),
                               instructions=self.instructions.text().strip(),
                               head_circumference=self.head.text().strip())
        RecordDialog.last = self.subject
        self.accept()
