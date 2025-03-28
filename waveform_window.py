"""
@file waveform_generator_window.py
@brief Main window for waveform generation.
@date 02/06/2025

Main window GUI for generating waveforms.

@author: John Rocco <jrocco@artemisinc.net>
"""

import io
from contextlib import redirect_stdout
from pathlib import Path
from queue import Empty, Queue

import yaml
from simulib.mesh_functions import readCombineMeshFile

from mesh_viewer import QGLControllerWidget
import torch
from PyQt5.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                             QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QSettings, QThread, QTimer
from PyQt5.QtOpenGL import QGLFormat, QGLWidget, QGL
import moderngl
from OpenGL.raw.GL.VERSION.GL_1_1 import *
from sdrparse import load
import numpy as np
from config import get_config
from models import TargetEmbedding
import os
import re
import sys

from contextlib import suppress
from waveform_model import GeneratorModel

fs = 2e9


# Waveform generator window
class WaveformGeneratorWindow(QMainWindow):
    patterns: list
    thread: QThread = None

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Waveform Generator")
        self.setGeometry(200, 200, 400, 300)
        # self.setFixedSize(600, 200)

        # Main layout
        layout = QVBoxLayout()

        # Grid layout for the widgets
        grid_layout = QGridLayout()
        layout.addLayout(grid_layout)

        # Target combo box and information
        # Load files in from ids.txt
        with open('./target_files.yaml', 'r') as file:
            try:
                self.target_scalings = list(yaml.safe_load(file).items())
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        with open('./data/target_ids.txt', 'r') as f:
            target_ids = [t.strip().split(":")[1][1:] for t in f.readlines()]
        self.target_files = [
            Path(f'/home/jeff/Documents/target_meshes/{t}') for t in target_ids]
        # Load mean tensors
        self.patterns = torch.load('./data/target_tensors/target_embedding_means.pt')
        grid_layout.addWidget(QLabel("Target:"), 0, 0)
        self.target_combo_box = QComboBox(self)
        grid_layout.addWidget(self.target_combo_box, 0, 1)
        self.target_combo_box.addItems(target_ids)
        self.target_combo_box.setStyleSheet("background-color: white;")
        self.target_combo_box.setEditable(True)
        self.target_combo_box.lineEdit().setAlignment(Qt.AlignCenter)
        self.target_combo_box.lineEdit().setReadOnly(True)
        self.target_combo_box.currentIndexChanged.connect(self.target_selected)
        for i in range(self.target_combo_box.count()):
            self.target_combo_box.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
        self.target_combo_box.setFixedWidth(100)

        # Pulse length spin box
        grid_layout.addWidget(QLabel("Pulse Length:"), 1, 0)
        self.pulse_length_spin_box = QDoubleSpinBox(self)
        grid_layout.addWidget(self.pulse_length_spin_box, 1, 1)
        self.pulse_length_spin_box.setRange(0.001, 100.0)
        self.pulse_length_spin_box.setDecimals(3)
        self.pulse_length_spin_box.setSingleStep(0.001)
        self.pulse_length_spin_box.setSuffix(" μs")
        self.pulse_length_spin_box.setFixedWidth(100)
        self.pulse_length_spin_box.lineEdit().setAlignment(Qt.AlignCenter)

        # Output folder line edit
        grid_layout.addWidget(QLabel("Output Folder:"), 2, 0)
        self.output_folder_line_edit = QLineEdit(self)
        self.output_folder_line_edit.setAcceptDrops(True)
        self.output_folder_line_edit.setReadOnly(True)
        grid_layout.addWidget(self.output_folder_line_edit, 2, 1)

        self.output_folder_browse_btn = QPushButton("Browse", self)
        self.output_folder_browse_btn.clicked.connect(self.browse_output_folder)
        grid_layout.addWidget(self.output_folder_browse_btn, 2, 2)

        # Background SAR file layout
        grid_layout.addWidget(QLabel("Background SAR File:"), 3, 0)
        self.sar_file_line_edit = QLineEdit(self)
        self.sar_file_line_edit.setAcceptDrops(True)
        self.sar_file_line_edit.setReadOnly(True)
        grid_layout.addWidget(self.sar_file_line_edit, 3, 1)

        self.sar_file_browse_btn = QPushButton("Browse", self)
        self.sar_file_browse_btn.clicked.connect(self.browse_sar_file)
        grid_layout.addWidget(self.sar_file_browse_btn, 3, 2)

        # Output file name layout
        grid_layout.addWidget(QLabel("Output File Name:"), 4, 0)
        self.output_file_line_edit = QLineEdit(self)
        grid_layout.addWidget(self.output_file_line_edit, 4, 1)
        # Connect the editingFinished signal to ensure_wave_suffix
        self.output_file_line_edit.editingFinished.connect(self.ensure_wave_suffix)

        # Load the persistent settings
        self.loadPersistentSettings()

        self.mview = QGLControllerWidget(self)
        grid_layout.addWidget(self.mview, 5, 0, 1, 3)

        # Start button and progress bar layout
        self.start_progress_layout = QHBoxLayout()
        layout.addLayout(self.start_progress_layout)

        # Start button
        self.start_button = QPushButton("Start", self)
        self.start_progress_layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setFixedWidth(100)

        # Progress bar
        self.progress_bar = ProgressBarWithText(self)
        self.start_progress_layout.addWidget(self.progress_bar)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setText('Loading...')

        # Set the central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        timer = QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.mview.updateGL)
        timer.start()



        self.progress_bar.setText('Setting up embedding model...')
        target_config = get_config('target_exp', './vae_config.yaml')
        embedding = TargetEmbedding.load_from_checkpoint(
            f'{target_config.weights_path}/{target_config.model_name}.ckpt',
            config=target_config, strict=False)
        self.progress_bar.setText('Setting up wavemodel...')
        model_config = get_config('wave_exp', './vae_config.yaml')
        self.wave_mdl = GeneratorModel.load_from_checkpoint(f'{model_config.weights_path}/{model_config.model_name}.ckpt',
                                                       config=model_config, embedding=embedding, strict=False)
        self.wave_mdl.eval()
        self.progress_bar.setText('Model loading complete.')

    # Browse for a SAR file
    def browse_sar_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select SAR File", "", "SAR Files (*.sar);;All Files (*)")
        if file_path:
            self.sar_file_line_edit.setText(file_path)

    # Browse for an output folder
    def browse_output_folder(self):
        if folder_path := QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        ):
            self.output_folder_line_edit.setText(folder_path)

    # Close event
    def closeEvent(self, a_event, **kwargs):
        self.savePersistentSettings()
        a_event.accept()

    def drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def ensure_wave_suffix(self):
        text = self.output_file_line_edit.text()
        if len(text) and not text.endswith('.wave'):
            self.output_file_line_edit.setText(f'{text}.wave')

    # Loads the persistent settings from registry
    def loadPersistentSettings(self):
        settings = QSettings("ARTEMIS", "WaveformGenerator")
        self.sar_file_line_edit.setText(settings.value("sar_file", ""))
        self.target_combo_box.setCurrentText(settings.value("target", ""))
        self.output_folder_line_edit.setText(settings.value("output_folder", ""))
        self.pulse_length_spin_box.setValue(float(settings.value("pulse_length_us", 0)))

    def target_selected(self, index):
        print(self.target_files[index])
        mesh = readCombineMeshFile(
            str(self.target_files[index]), 10000,
            scale=1 / self.target_scalings[index][1])
        self.mview.set_mesh(mesh)

    def sar_file_drop_event(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith(".sar"):
                self.sar_file_line_edit.setText(file_path)
                return
        self.show_message("Please drop a valid .sar file.")

    def output_folder_drop_event(self, event):
        for url in event.mimeData().urls():
            folder_path = url.toLocalFile()
            if os.path.isdir(folder_path):
                self.output_folder_line_edit.setText(folder_path)
                return
        self.show_message("Please drop a valid folder.")

    # Saves the persistent settings to registry
    def savePersistentSettings(self):
        settings = QSettings("ARTEMIS", "WaveformGenerator")
        settings.setValue("sar_file", self.sar_file_line_edit.text())
        settings.setValue("target", self.target_combo_box.currentText())
        settings.setValue("output_folder", self.output_folder_line_edit.text())
        settings.setValue("pulse_length_us", self.pulse_length_spin_box.value())

    def show_message(self, a_message):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(a_message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    # Update the progress bar
    def slot_update_progress(self, value):
        self.progress_bar.setText(value)

    def slot_update_percentage(self, value):
        self.progress_bar.setValue(value)

    # Start processing
    def start_processing(self):
        self.toggleGUIElements(False)
        target = int(self.target_combo_box.currentText().split(':')[0])
        pulse_length = self.pulse_length_spin_box.value() * 1e-6
        sar_file = self.sar_file_line_edit.text()
        output_folder = self.output_folder_line_edit.text()
        output_file = self.output_file_line_edit.text()

        if not sar_file:
            self.show_message("Please specify a SAR file.")
            return

        if not output_folder:
            self.show_message("Please specify an output folder.")
            return

        self.thread = GenerateWaveformThread(self.patterns[target], pulse_length, sar_file, f'{output_folder}/{output_file}', self.wave_mdl)
        self.thread.signal_update_progress.connect(self.slot_update_progress)
        self.thread.signal_update_percentage.connect(self.slot_update_percentage)
        self.thread.finished.connect(lambda: self.toggleGUIElements(True))
        self.thread.start()

    # Enable or disable GUI elements
    def toggleGUIElements(self, enabled):
        self.target_combo_box.setEnabled(enabled)
        self.pulse_length_spin_box.setEnabled(enabled)
        self.sar_file_line_edit.setEnabled(enabled)
        self.sar_file_browse_btn.setEnabled(enabled)
        self.output_folder_line_edit.setEnabled(enabled)
        self.output_folder_browse_btn.setEnabled(enabled)
        self.output_file_line_edit.setEnabled(enabled)
        self.start_button.setEnabled(enabled)

# Generate waveform thread
class GenerateWaveformThread(QThread):
    signal_update_progress = pyqtSignal(str)
    signal_update_percentage = pyqtSignal(int)
    def __init__(self, target, pulse_length, sar_file, output_path, model):
        super().__init__()
        self.target = target
        self.pulse_length = pulse_length
        self.sar_file = sar_file
        self.output_path = output_path
        self.model = model

    def run(self):
        self.signal_update_progress.emit('Loading SDR file...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sdr = load(self.sar_file, progress_tracker=True)
        self.signal_update_percentage.emit(25)
        self.signal_update_progress.emit('Loaded SDR file.')
        pulses = np.fft.fft(sdr.getPulses(sdr[0].frame_num[:32], 0)[1].T, self.model.fft_len, axis=-1)
        mfilt = sdr.genMatchedFilter(0, fft_len=self.model.fft_len)
        pulse_data = np.fft.fftshift(pulses * mfilt)
        pulse_bw = 250e6
        nr = int(self.pulse_length * fs)

        self.model.to(device)
        waves = self.model.full_forward(pulse_data, torch.tensor(self.target,
                                                                 dtype=torch.float32).squeeze(0).to(device),
                                        nr, pulse_bw / fs)
        self.signal_update_progress.emit('Waveform generated.')
        self.signal_update_percentage.emit(50)
        self.signal_update_progress.emit('Saving file...')

        upsampled_waves = np.zeros((self.model.fft_len * 2,), dtype=np.complex64)
        upsampled_waves[:self.model.fft_len // 2] = waves[0, :self.model.fft_len // 2]
        upsampled_waves[-self.model.fft_len // 2:] = waves[0, -self.model.fft_len // 2:]
        new_fc = 250e6 + pulse_bw / 2
        self.signal_update_percentage.emit(70)

        bin_shift = int(np.round(new_fc / (fs / self.model.fft_len)))
        upsampled_waves = np.roll(upsampled_waves, bin_shift)
        time_wave = np.fft.ifft(upsampled_waves)[:nr * 2]
        scaling = max(time_wave.real.max(), abs(time_wave.real.min()))
        output_wave = (time_wave.real / scaling).astype(np.float32)
        self.signal_update_percentage.emit(80)
        try:
            with open(self.output_path, 'wb') as f:
                final = np.concatenate((np.array([new_fc, pulse_bw], dtype=np.float32), output_wave))
                final.tofile(f)
                self.signal_update_progress[str].emit(f'File saved as {self.output_path}.')
                self.signal_update_percentage.emit(0)
        except IOError as e:
            print(f'Error writing to file. {e}')


class ProgressBarWithText(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft)

    def setText(self, text):
        self.setFormat(text)


class ObjViewer(QGLWidget):
    def __init__(self, filename, parent=None):
        fmt = QGLFormat()
        fmt.setVersion(3, 2)
        fmt.setProfile(QGLFormat.CoreProfile)
        QGLFormat.setDefaultFormat(fmt)
        super().__init__(fmt, parent)
        self.vertices = []
        self.model_origin = [0, 0, 0.]
        self.faces = []
        self.load_obj(filename)
        self.paintGL()

    def load_obj(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertex = list(map(float, line.split()[1:]))
                    self.vertices.append(vertex)
                    self.model_origin = [m + v for m, v in zip(self.model_origin, vertex)]
                elif line.startswith('f '):
                    face = [int(idx.split('/')[0]) for idx in line.split()[1:]]
                    self.faces.append(face)
        self.model_origin = [m / len(self.vertices) for m in self.model_origin]

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 200, 200, 0, -1, 1)


    def paintGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBegin(GL_TRIANGLES)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(-self.model_origin[0], -self.model_origin[1], -self.model_origin[2])
        for face in self.faces:
            for vertex_index in face:
                glColor3fv([0.8, 0.3, 0.3])  # Red color
                vertex = self.vertices[vertex_index - 1]
                glVertex3fv(vertex)
        glEnd()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

# Main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WaveformGeneratorWindow()
    window.show()
    sys.exit(app.exec_())