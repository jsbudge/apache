from PyQt5.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                             QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QSettings, QThread, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from scipy.spatial.transform import Rotation as R
from simulib.simulation_functions import genChirp, db
from superqt import QLargeIntSpinBox


class ProgressBarWithText(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft)

    def setText(self, text):
        self.setFormat(text)


class MplWidget(QWidget):

    def __init__(self, parent=None, figsize=(5, 5)):
        super().__init__(parent)

        fig = Figure(figsize=figsize)
        self.can = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.can, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.can)

        # here you can set up your figure/axis
        self.ax = None

    def plot_basic_line(self, X, Y, label):
        # plot a basic line plot from x and y values.
        self.ax = self.can.figure.add_axes([.1, .2, .8, .6])
        self.ax.cla()  # clears the axis
        self.ax.plot(X, Y, label=label)
        self.ax.grid(True)
        # self.ax.legend()
        self.ax.set_ylabel('Power (dB)')
        self.ax.set_xlabel('Frequency (MHz)')
        # self.can.figure.tight_layout()
        self.can.draw()

    def plot_dopp_map(self, data):
        self.ax = self.can.figure.add_subplot(111)
        self.ax.cla()
        self.ax.imshow(data, aspect='auto')
        self.ax.axis('off')
        self.can.figure.tight_layout()
        self.can.draw()


class FileSelectWidget(QWidget):
    signal_btn_clicked = pyqtSignal(str)

    def __init__(self, parent=None, label_name='File:', file_types=None, read_only=True):
        super().__init__(parent)
        layout = QHBoxLayout()

        layout.addWidget(QLabel(label_name))
        self.line_edit = QLineEdit(self)
        self.line_edit.setAcceptDrops(True)
        self.line_edit.setReadOnly(read_only)

        self.browse_btn = QPushButton("Browse", self)
        self.browse_btn.clicked.connect(self.browse_output_folder)

        self.file_types = file_types

        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_btn)
        self.setLayout(layout)

    def browse_output_folder(self):
        if self.file_types is None:
            if _path := QFileDialog.getExistingDirectory(
                    self, "Select Output Folder"
            ):
                self.line_edit.setText(_path)
        else:
            _path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self.file_types)
        self.line_edit.setText(_path)
        self.signal_btn_clicked.emit(_path)


class WaveformCreateWindow(QMainWindow):
    signal_add_wave = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waveform Creator")
        self.setGeometry(250, 250, 500, 500)

        self.wave = genChirp(4300, 2e9, 9.6e9, 150e6)
        layout = QVBoxLayout()

        self.time_view = MplWidget(self)
        self.freq_view = MplWidget(self)
        opts_layout = QHBoxLayout()
        add_layout = QHBoxLayout()
        self.bandwidth_spinbox = QDoubleSpinBox()
        self.bandwidth_spinbox.setValue(.5)
        self.bandwidth_spinbox.setRange(.01, .95)
        self.nsam_spinbox = QLargeIntSpinBox()
        self.nsam_spinbox.setValue(4300)
        self.nsam_spinbox.setRange(1000, 8000)
        self.fc_spinbox = QDoubleSpinBox()
        self.fc_spinbox.setValue(9.6e9)
        self.fc_spinbox.setRange(8e9, 10e9)
        self.bandwidth_spinbox.valueChanged.connect(self.slot_gen_wave)
        self.nsam_spinbox.valueChanged.connect(self.slot_gen_wave)
        self.fc_spinbox.valueChanged.connect(self.slot_gen_wave)
        self.add_button = QPushButton('Add to Simulation')
        self.add_button.clicked.connect(self.slot_gen_wave)
        self.name_field = QLineEdit(self)

        opts_layout.addWidget(QLabel('BW:'))
        opts_layout.addWidget(self.bandwidth_spinbox)
        opts_layout.addWidget(QLabel('NSAM:'))
        opts_layout.addWidget(self.nsam_spinbox)
        opts_layout.addWidget(QLabel('FC:'))
        opts_layout.addWidget(self.fc_spinbox)

        add_layout.addWidget(self.add_button)
        add_layout.addWidget(self.name_field)

        layout.addWidget(self.time_view)
        layout.addWidget(self.freq_view)
        layout.addLayout(opts_layout)
        layout.addLayout(add_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.slot_gen_wave()

    def slot_gen_wave(self):
        self.wave = genChirp(self.nsam_spinbox.value(), 2e9, self.fc_spinbox.value(), self.bandwidth_spinbox.value() * 2e9)
        freq_wave = np.fft.fft(self.wave, 8192)
        freqs = np.fft.fftfreq(len(freq_wave), 1 / 2e9)
        self.time_view.plot_basic_line(np.arange(self.wave.shape[0]), self.wave.real, 'Time Series')
        self.freq_view.plot_basic_line(freqs, db(freq_wave), 'Frequency')

    def slot_add_wave(self):
        self.slot_gen_wave()
        self.signal_add_wave.emit([self.name_field.text(), self.wave])
