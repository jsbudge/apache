"""
@file waveform_generator_window.py
@brief Main window for waveform generation.
@date 02/06/2025

Main window GUI for generating waveforms.

@author: John Rocco <jrocco@artemisinc.net>
"""

from pathlib import Path
import yaml
from sdrparse.SDRParsing import SDRBase
from simulib.mesh_functions import readCombineMeshFile
from simulib.simulation_functions import db
from gui_classes import ProgressBarWithText, MplWidget
from mesh_viewer import QGLControllerWidget
import torch
from PyQt5.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                             QLineEdit, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget, QSpinBox,
                             QSplashScreen, QCheckBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, Qt, QSettings, QThread, QTimer
from superqt import QRangeSlider
from sdrparse import load
import numpy as np
from config import get_config
from models import TargetEmbedding
import os
import sys
from waveform_model import GeneratorModel

fs = 2e9


# Waveform generator window
class WaveformGeneratorWindow(QMainWindow):
    patterns: list
    thread: QThread = None
    sdr: SDRBase = None
    win_width: int = 500
    win_height: int = 500
    win_full_width: int = 1200

    def __init__(self, model):
        super().__init__()

        self.setWindowTitle("Waveform Generator")
        self.setGeometry(200, 200, self.win_width, self.win_height)
        self.setFixedSize(self.win_width, self.win_height)

        # Main layout
        main_layout = QHBoxLayout()
        settings_layout = QVBoxLayout()
        plotting_layout = QVBoxLayout()

        # Grid layout for the widgets
        grid_layout = QGridLayout()
        settings_layout.addLayout(grid_layout)

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
        grid_layout.addWidget(self.target_combo_box, 0, 1, 1, 2)
        self.target_combo_box.addItems(target_ids)
        self.target_combo_box.setStyleSheet("background-color: white;")
        self.target_combo_box.setEditable(True)
        self.target_combo_box.lineEdit().setAlignment(Qt.AlignCenter)
        self.target_combo_box.lineEdit().setReadOnly(True)
        self.target_combo_box.currentIndexChanged.connect(self.target_selected)
        for i in range(self.target_combo_box.count()):
            self.target_combo_box.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
        self.target_combo_box.setFixedWidth(300)

        # Pulse length spin box
        grid_layout.addWidget(QLabel("Pulse Length:"), 1, 0)
        self.pulse_length_spin_box = QDoubleSpinBox(self)
        grid_layout.addWidget(self.pulse_length_spin_box, 1, 1)
        self.pulse_length_spin_box.setRange(0.001, 5.0)
        self.pulse_length_spin_box.setDecimals(3)
        self.pulse_length_spin_box.setSingleStep(0.001)
        self.pulse_length_spin_box.setSuffix(" μs")
        self.pulse_length_spin_box.setFixedWidth(100)
        self.pulse_length_spin_box.lineEdit().setAlignment(Qt.AlignCenter)

        # Bandwidth spin box
        grid_layout.addWidget(QLabel("Bandwidth:"), 1, 2)
        self.bandwidth_spin_box = QDoubleSpinBox(self)
        grid_layout.addWidget(self.bandwidth_spin_box, 1, 3)
        self.bandwidth_spin_box.setRange(250., 1600.0)
        self.bandwidth_spin_box.setDecimals(1)
        self.bandwidth_spin_box.setSingleStep(10.)
        self.bandwidth_spin_box.setSuffix(" MHz")
        self.bandwidth_spin_box.setFixedWidth(100)
        self.bandwidth_spin_box.lineEdit().setAlignment(Qt.AlignCenter)

        # Output folder line edit
        grid_layout.addWidget(QLabel("Output Folder:"), 2, 0)
        self.output_folder_line_edit = QLineEdit(self)
        self.output_folder_line_edit.setAcceptDrops(True)
        self.output_folder_line_edit.setReadOnly(True)
        grid_layout.addWidget(self.output_folder_line_edit, 2, 1, 1, 2)

        self.output_folder_browse_btn = QPushButton("Browse", self)
        self.output_folder_browse_btn.clicked.connect(self.browse_output_folder)
        grid_layout.addWidget(self.output_folder_browse_btn, 2, 3)

        # Background SAR file layout
        grid_layout.addWidget(QLabel("Background SAR File:"), 3, 0)
        self.sar_file_line_edit = QLineEdit(self)
        self.sar_file_line_edit.setAcceptDrops(True)
        self.sar_file_line_edit.setReadOnly(True)
        grid_layout.addWidget(self.sar_file_line_edit, 3, 1, 1, 2)

        self.sar_file_browse_btn = QPushButton("Browse", self)
        self.sar_file_browse_btn.clicked.connect(self.browse_sar_file)
        grid_layout.addWidget(self.sar_file_browse_btn, 3, 3)

        # Selection of pulses for background
        grid_layout.addWidget(QLabel("Pulse Range"), 4, 0)
        self.sar_pulse_range = QRangeSlider(Qt.Orientation.Horizontal)
        self.pulse_display = QLabel('1-33')
        self.range_values = (1, 33)
        self.sar_pulse_range.setRange(1, 1000)
        self.sar_pulse_range.setValue(self.range_values)
        self.sar_pulse_range.valuesChanged.connect(self.slot_rigidize_bar)
        self.sar_pulse_range.sliderReleased.connect(lambda: [self.sar_pulse_range.setValue(self.range_values),
                                                    self.pulse_display.setText(f'{self.range_values[0]}-{self.range_values[1]}')])
        grid_layout.addWidget(self.sar_pulse_range, 4, 1, 1, 2)
        grid_layout.addWidget(self.pulse_display, 4, 3)

        grid_layout.addWidget(QLabel("Range Size:"), 5, 0)
        self.range_size_spin_box = QSpinBox(self)
        self.range_size_spin_box.setRange(32, 128)
        self.range_size_spin_box.setSingleStep(32)
        self.range_size_spin_box.valueChanged.connect(lambda: [self.sar_pulse_range.setValue(self.range_values),
                                                             self.pulse_display.setText(
                                                                 f'{self.range_values[0]}-{self.range_values[1]}')])
        grid_layout.addWidget(self.range_size_spin_box, 5, 1)
        grid_layout.addWidget(QLabel("Show RDA Map"), 5, 2)
        self.show_rdmap_check_box = QCheckBox(self)
        self.show_rdmap_check_box.clicked.connect(self.slot_checkbox)
        grid_layout.addWidget(self.show_rdmap_check_box, 5, 3)

        # Output file name layout
        grid_layout.addWidget(QLabel("Output File Name:"), 6, 0)
        self.output_file_line_edit = QLineEdit(self)
        grid_layout.addWidget(self.output_file_line_edit, 6, 1, 1, 2)
        # Connect the editingFinished signal to ensure_wave_suffix
        self.output_file_line_edit.editingFinished.connect(self.ensure_wave_suffix)

        # Start button and progress bar layout
        self.start_progress_layout = QHBoxLayout()
        settings_layout.addLayout(self.start_progress_layout)

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

        # Load the persistent settings
        self.loadPersistentSettings()

        self.mview = QGLControllerWidget(self)
        grid_layout.addWidget(self.mview, 7, 0, 1, 4)
        self.mview.updateGL()

        # Matplotlib waveform and sar selection
        self.plot_window = MplWidget(self, figsize=(25, 150))
        self.plot_window.setVisible(False)
        plotting_layout.addWidget(self.plot_window)

        # Pulse Selection doppler window
        self.sar_window = MplWidget(self, figsize=(150, 150))
        self.sar_window.setVisible(False)
        plotting_layout.addWidget(self.sar_window)


        # Set the central widget
        main_layout.addLayout(settings_layout)
        main_layout.addLayout(plotting_layout)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        timer = QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.mview.updateGL)
        timer.start()

        self.wave_mdl = model
        self.progress_bar.setText('Model loading complete.')

    # Browse for a SAR file
    def browse_sar_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select SAR File", "", "SAR Files (*.sar);;All Files (*)")
        if file_path:
            self.load_sar_file(file_path)

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

    def slot_rigidize_bar(self, values):
        if values[0] < self.range_values[0]:
            self.range_values = (values[0], values[0] + self.range_size_spin_box.value())
        elif values[1] > self.range_values[1]:
            self.range_values = (values[1] - self.range_size_spin_box.value(), values[1])
        else:
            self.range_values = (int(values[0] - self.range_size_spin_box.value() / 2),
                                 int(values[1] + self.range_size_spin_box.value() / 2))
            self.range_values = (0, self.range_size_spin_box.value()) if self.range_values[0] <= 0 else self.range_values
        # self.sar_pulse_range.setValue(self.range_values)

    def slot_checkbox(self, value):
        pass


    def ensure_wave_suffix(self):
        text = self.output_file_line_edit.text()
        if len(text) and not text.endswith('.wave'):
            self.output_file_line_edit.setText(f'{text}.wave')

    # Loads the persistent settings from registry
    def loadPersistentSettings(self):
        settings = QSettings("ARTEMIS", "WaveformGenerator")
        self.load_sar_file(settings.value("sar_file", ""))
        self.target_combo_box.setCurrentText(settings.value("target", ""))
        self.output_folder_line_edit.setText(settings.value("output_folder", ""))
        self.pulse_length_spin_box.setValue(float(settings.value("pulse_length_us", 0)))
        self.bandwidth_spin_box.setValue(float(settings.value("bandwidth_mhz", 0)))

    def target_selected(self, index):
        mesh = readCombineMeshFile(
            str(self.target_files[index]), 10000,
            scale=1 / self.target_scalings[index][1])
        self.mview.set_mesh(mesh)

    def sar_file_drop_event(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith(".sar"):
                self.load_sar_file(file_path)
                return
        self.show_message("Please drop a valid .sar file.")

    def load_sar_file(self, fnme):
        self.sar_file_line_edit.setText(fnme)
        self.progress_bar.setText('Loading .sar file...')
        self.sdr = load(fnme, progress_tracker=True)
        self.sar_pulse_range.setRange(self.sdr[0].frame_num[0], self.sdr[0].frame_num[-1])
        self.progress_bar.setText('Loaded SDR file.')


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
        settings.setValue("bandwidth_mhz", self.bandwidth_spin_box.value())

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
        target = int(self.target_combo_box.currentIndex())
        pulse_length = self.pulse_length_spin_box.value() * 1e-6
        bandwidth = self.bandwidth_spin_box.value() * 1e6
        sar_file = self.sdr
        output_folder = self.output_folder_line_edit.text()
        output_file = self.output_file_line_edit.text()

        if not sar_file:
            self.show_message("Please specify a SAR file.")
            return

        if not output_folder:
            self.show_message("Please specify an output folder.")
            return

        if not output_file:
            self.show_message("Please specify an output filename.")
            return

        self.toggleGUIElements(False)

        self.thread = GenerateWaveformThread(self.patterns[target], pulse_length, bandwidth, sar_file,
                                             self.sar_pulse_range.value()[0], self.sar_pulse_range.value()[1],
                                             f'{output_folder}/{output_file}', self.wave_mdl)
        self.thread.signal_update_progress.connect(self.slot_update_progress)
        self.thread.signal_update_percentage.connect(self.slot_update_percentage)
        self.thread.signal_waveform_generated.connect(self.slot_updatePlot)
        self.thread.finished.connect(lambda: self.toggleGUIElements(True))
        self.thread.start()

    # Enable or disable GUI elements
    def toggleGUIElements(self, enabled):
        self.target_combo_box.setEnabled(enabled)
        self.pulse_length_spin_box.setEnabled(enabled)
        self.bandwidth_spin_box.setEnabled(enabled)
        self.sar_file_line_edit.setEnabled(enabled)
        self.sar_file_browse_btn.setEnabled(enabled)
        self.output_folder_line_edit.setEnabled(enabled)
        self.output_folder_browse_btn.setEnabled(enabled)
        self.output_file_line_edit.setEnabled(enabled)
        self.start_button.setEnabled(enabled)

    def slot_updatePlot(self, wavedata):
        self.progress_bar.setText('Updating plots...')
        plot_waves = db(wavedata[0])
        plot_freqs = np.fft.fftfreq(wavedata[0].shape[-1])
        self.plot_window.plot_basic_line(np.fft.fftshift(plot_freqs), np.fft.fftshift(plot_waves[0]), 'waveform')
        self.plot_window.setVisible(True)

        dopp_wave = np.fft.fftshift(db(wavedata[1]).T)
        self.sar_window.plot_dopp_map(dopp_wave)
        self.sar_window.setVisible(True)

        self.setFixedSize(self.win_full_width, self.win_height)

# Generate waveform thread
class GenerateWaveformThread(QThread):
    signal_update_progress = pyqtSignal(str)
    signal_update_percentage = pyqtSignal(int)
    signal_waveform_generated = pyqtSignal(object)

    def __init__(self, target, pulse_length, bandwidth, sar_file, frame_min, frame_max, output_path, model):
        super().__init__()
        self.target = target
        self.pulse_length = pulse_length
        self.bandwidth = bandwidth
        self.sar_file = sar_file
        self.output_path = output_path
        self.model = model
        self.frame_range = (frame_min, frame_max)

    def run(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sdr = self.sar_file
        self.signal_update_percentage.emit(25)
        self.signal_update_progress.emit('Loaded SDR file. Getting pulses...')
        pulses = np.fft.fft(sdr.getPulses(sdr[0].frame_num[self.frame_range[0]:self.frame_range[1]], 0)[1].T,
                            self.model.fft_len, axis=-1)
        self.signal_update_progress.emit('Generating matched filter...')
        self.signal_update_percentage.emit(35)
        mfilt = sdr.genMatchedFilter(0, fft_len=self.model.fft_len)
        pdata = pulses * mfilt
        pulse_data = np.fft.fftshift(pdata)
        nr = int(self.pulse_length * fs)

        self.signal_update_progress.emit('Moving model to GPU...')
        self.signal_update_percentage.emit(40)
        self.model.to(device)
        self.signal_update_progress.emit('Generating waveforms...')
        waves = self.model.full_forward(pulse_data, torch.tensor(self.target,
                                                                 dtype=torch.float32).squeeze(0).to(device),
                                        nr, self.bandwidth / fs)
        self.signal_update_progress.emit('Waveforms generated.')
        self.signal_update_percentage.emit(70)
        self.signal_update_progress.emit('Saving file...')

        upsampled_waves = np.zeros((self.model.fft_len * 2,), dtype=np.complex64)
        upsampled_waves[:self.model.fft_len // 2] = waves[0, :self.model.fft_len // 2]
        upsampled_waves[-self.model.fft_len // 2:] = waves[0, -self.model.fft_len // 2:]
        new_fc = 250e6 + self.bandwidth / 2
        self.signal_update_percentage.emit(80)

        bin_shift = int(np.round(new_fc / (fs / self.model.fft_len)))
        upsampled_waves = np.roll(upsampled_waves, bin_shift)
        time_wave = np.fft.ifft(upsampled_waves)[:nr * 2]
        scaling = max(time_wave.real.max(), abs(time_wave.real.min()))
        output_wave = (time_wave.real / scaling).astype(np.float32)
        self.signal_update_percentage.emit(90)
        try:
            with open(self.output_path, 'wb') as f:
                final = np.concatenate((np.array([new_fc, self.bandwidth], dtype=np.float32), output_wave))
                final.tofile(f)
                self.signal_update_progress[str].emit(f'File saved as {self.output_path}.')
                self.signal_update_percentage.emit(0)
        except IOError as e:
            print(f'Error writing to file. {e}')
        doppler_wave = np.fft.fft(np.fft.ifft(pdata, axis=-1), axis=0)
        self.signal_waveform_generated.emit((waves, doppler_wave))


# Main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    spl_pix = QPixmap('./artemislogo.png')
    splash = QSplashScreen(spl_pix, Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage('Loading model configuration files...')
    target_config = get_config('target_exp', './vae_config.yaml')
    splash.showMessage('Loading embedding model...')
    embedding = TargetEmbedding.load_from_checkpoint(
        f'{target_config.weights_path}/{target_config.model_name}.ckpt',
        config=target_config, strict=False)
    splash.showMessage('Loading wavemodel configuration files...')
    model_config = get_config('wave_exp', './vae_config.yaml')
    splash.showMessage('Loading wavemodel...')
    wave_mdl = GeneratorModel.load_from_checkpoint(f'{model_config.weights_path}/{model_config.model_name}.ckpt',
                                                        config=model_config, embedding=embedding, strict=False)
    wave_mdl.eval()
    splash.showMessage('Done.')
    splash.showMessage('Building interface...')
    window = WaveformGeneratorWindow(wave_mdl)
    splash.close()
    window.show()
    sys.exit(app.exec_())