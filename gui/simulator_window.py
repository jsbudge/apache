from functools import partial

import pandas as pd
import torch
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import QThread, QSettings, pyqtSignal
from PyQt5.QtGui import QIcon
from OpenGL.GLUT import *
from PyQt5.QtWidgets import QAction, QHBoxLayout, QVBoxLayout, QGridLayout, QWidget, QDoubleSpinBox, QLabel, \
    QRadioButton, QButtonGroup, QPushButton, QComboBox
from sdrparse.SDRParsing import SDRBase, load
from simulib.grid_helper import SDREnvironment
from simulib.mesh_functions import readCombineMeshFile
from simulib.platform_helper import SDRPlatform
from simulib.simulation_functions import genChirp
from sklearn.decomposition import TruncatedSVD
from superqt import QLargeIntSpinBox
from utils import get_radar_coeff
from numba import cuda
from gui.gui_classes import FileSelectWidget, ProgressBarWithText
from mesh_viewer import QGLControllerWidget, ball
import gui_utils as f
import argparse
import numpy as np

from target_data_generator import loadClutterTargetSpectrum, getTargetProfile, processTargetProfile, genProfileFromMesh


class MainWindow(QtWidgets.QMainWindow):
    patterns: list
    thread: QThread = None
    _rp: SDRPlatform = None
    _bg: SDREnvironment = None
    elevation_map: np.array = None
    virtual_pos: np.array = np.array([0., 0., 0.])
    radar_coeff: float = 0.
    _ntris: int = 10000
    _scaling: float = 7.
    win_width: int = 500
    win_height: int = 500
    win_full_width: int = 1200
    _model_fnme: str = None
    _mesh_path: str = None
    _mode: str = 'Profile'

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        self.setWindowTitle("Simulator")
        self.setGeometry(200, 200, self.win_width, self.win_height)
        # self.setFixedSize(self.win_width, self.win_height)
        # menu_bar = self.menuBar()
        # new_window_action = QAction("Settings", self)
        # new_window_action.triggered.connect(self.open_settings_window)
        # menu_bar.addAction(new_window_action)

        self.target_info = pd.read_csv('../data/target_info.csv')

        main_layout = QGridLayout()

        # Create openGL context
        self.openGL = QGLControllerWidget(self)
        self.openGL.setFixedSize(500, 500)
        main_layout.addWidget(self.openGL, 0, 0, 15, 3)
        self.simulate_button = QPushButton('Simulate!')
        self.simulate_button.clicked.connect(self.run_simulation)

        self.target_combo_box = QComboBox(self)
        self.target_combo_box.addItems(self.target_info['name'])
        # self.target_combo_box.lineEdit().setReadOnly(True)
        self.target_combo_box.currentIndexChanged.connect(self.loadMesh)

        self.sdr_file = FileSelectWidget(self, 'Select SDR File', file_types="SAR Files (*.sar)")
        self.sdr_file.signal_btn_clicked.connect(self.loadSAR)

        position_layout = QGridLayout()

        self.x_pos_spinbox = QDoubleSpinBox(self)
        self.x_pos_spinbox.setRange(-10000., 10000.)
        self.x_pos_spinbox.setFixedWidth(70)
        self.y_pos_spinbox = QDoubleSpinBox(self)
        self.y_pos_spinbox.setRange(-10000., 10000.)
        self.y_pos_spinbox.setFixedWidth(70)
        self.z_pos_spinbox = QDoubleSpinBox(self)
        self.z_pos_spinbox.setRange(-10000., 10000.)
        self.z_pos_spinbox.setFixedWidth(70)
        self.x_pos_spinbox.valueChanged.connect(self.setPosition)
        self.y_pos_spinbox.valueChanged.connect(self.setPosition)
        self.z_pos_spinbox.valueChanged.connect(self.setPosition)
        self.azimuth_spinbox = QLargeIntSpinBox(self)
        self.azimuth_spinbox.setValue(32)
        self.elevation_spinbox = QLargeIntSpinBox(self)
        self.elevation_spinbox.setValue(32)
        self.range_spinbox = QDoubleSpinBox(self)
        self.range_spinbox.setValue(500.)
        self.range_spinbox.setRange(1., 25000.)
        self.azimuth_spinbox.valueChanged.connect(self.updateGrid)
        self.elevation_spinbox.valueChanged.connect(self.updateGrid)
        self.range_spinbox.valueChanged.connect(self.updateGrid)
        position_layout.addWidget(self.x_pos_spinbox, 0, 1)
        position_layout.addWidget(self.y_pos_spinbox, 0, 3)
        position_layout.addWidget(self.z_pos_spinbox, 0, 5)
        position_layout.addWidget(self.azimuth_spinbox, 1, 1)
        position_layout.addWidget(self.elevation_spinbox, 1, 3)
        position_layout.addWidget(self.range_spinbox, 1, 5)
        position_layout.addWidget(QLabel('X:'), 0, 0)
        position_layout.addWidget(QLabel('Y:'), 0, 2)
        position_layout.addWidget(QLabel('Z:'), 0, 4)
        position_layout.addWidget(QLabel('# Az:'), 1, 0)
        position_layout.addWidget(QLabel('# El:'), 1, 2)
        position_layout.addWidget(QLabel('Range:'), 1, 4)

        self.mode_target = QRadioButton('Profile')
        self.mode_train = QRadioButton('SDR Train')
        self.mode_buttons = QButtonGroup()
        self.mode_buttons.addButton(self.mode_target)
        self.mode_buttons.addButton(self.mode_train)
        self.mode_buttons.buttonToggled.connect(self.setMode)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.mode_target)
        mode_layout.addWidget(self.mode_train)

        mesh_param_layout = QHBoxLayout()
        self.triangle_spinbox = QLargeIntSpinBox()
        self.triangle_spinbox.setRange(10, 5e6)
        self.triangle_spinbox.valueChanged.connect(self.loadMesh)
        self.scaling_spinbox = QDoubleSpinBox()
        self.scaling_spinbox.setRange(.01, 1000)
        self.scaling_spinbox.valueChanged.connect(self.loadMesh)
        mesh_param_layout.addWidget(QLabel('# Tris:'))
        mesh_param_layout.addWidget(self.triangle_spinbox)
        mesh_param_layout.addWidget(QLabel('Scaling Factor:'))
        mesh_param_layout.addWidget(self.scaling_spinbox)

        sdr_param_layout = QHBoxLayout()
        self.grid_width = QDoubleSpinBox()
        self.grid_width.setRange(1., 100.)
        self.grid_height = QDoubleSpinBox()
        self.grid_height.setRange(1., 100.)
        self.grid_nrows = QLargeIntSpinBox()
        self.grid_nrows.setRange(10, 100)
        self.grid_ncols = QLargeIntSpinBox()
        self.grid_ncols.setRange(10, 100)
        self.grid_width.valueChanged.connect(self.loadElevationMap)
        self.grid_height.valueChanged.connect(self.loadElevationMap)
        self.grid_nrows.valueChanged.connect(self.loadElevationMap)
        self.grid_ncols.valueChanged.connect(self.loadElevationMap)
        sdr_param_layout.addWidget(QLabel('Width:'))
        sdr_param_layout.addWidget(self.grid_width)
        sdr_param_layout.addWidget(QLabel('Height:'))
        sdr_param_layout.addWidget(self.grid_height)
        sdr_param_layout.addWidget(QLabel('Rows:'))
        sdr_param_layout.addWidget(self.grid_nrows)
        sdr_param_layout.addWidget(QLabel('Cols:'))
        sdr_param_layout.addWidget(self.grid_ncols)

        # File saving
        self.target_save_path = FileSelectWidget(self, 'Target Profile Save Path',
                                          read_only=False)
        self.clutter_save_path = FileSelectWidget(self, 'SAR Train Save Path',
                                                 read_only=False)

        # Radar parameters
        rparam_layout = QHBoxLayout()
        rparam_sec_layout = QHBoxLayout()
        self.fc_spinbox = QDoubleSpinBox()
        self.fc_spinbox.setValue(9600)
        self.fc_spinbox.setRange(8000, 10000)
        rparam_layout.addWidget(QLabel('FC (MHz):'))
        rparam_layout.addWidget(self.fc_spinbox)
        self.tx_power_spinbox = QDoubleSpinBox()
        self.tx_power_spinbox.setValue(50)
        self.tx_power_spinbox.setRange(10, 100)
        rparam_layout.addWidget(QLabel('Tx Power (dB):'))
        rparam_layout.addWidget(self.tx_power_spinbox)
        self.tx_gain_spinbox = QDoubleSpinBox()
        self.tx_gain_spinbox.setValue(1)
        self.tx_gain_spinbox.setRange(1, 500)
        rparam_sec_layout.addWidget(QLabel('Tx Gain (dB):'))
        rparam_sec_layout.addWidget(self.tx_gain_spinbox)
        self.rx_gain_spinbox = QDoubleSpinBox()
        self.rx_gain_spinbox.setValue(1)
        self.rx_gain_spinbox.setRange(1, 500)
        rparam_sec_layout.addWidget(QLabel('Rx Gain (dB):'))
        rparam_sec_layout.addWidget(self.rx_gain_spinbox)
        self.rec_gain_spinbox = QDoubleSpinBox()
        self.rec_gain_spinbox.setValue(1)
        self.rec_gain_spinbox.setRange(1, 500)
        rparam_layout.addWidget(QLabel('Rec Gain (dB):'))
        rparam_layout.addWidget(self.rec_gain_spinbox)
        self.fc_spinbox.valueChanged.connect(self.slot_set_radar_coeff)
        self.tx_power_spinbox.valueChanged.connect(self.slot_set_radar_coeff)
        self.tx_gain_spinbox.valueChanged.connect(self.slot_set_radar_coeff)
        self.rx_gain_spinbox.valueChanged.connect(self.slot_set_radar_coeff)
        self.rec_gain_spinbox.valueChanged.connect(self.slot_set_radar_coeff)

        # Simulation Iteration
        self.iteration_spinbox = QLargeIntSpinBox()
        self.iteration_spinbox.setValue(5)
        self.iteration_spinbox.setRange(1, 100)
        rparam_sec_layout.addWidget(QLabel('# Files to Generate:'))
        rparam_sec_layout.addWidget(self.iteration_spinbox)



        # Progress Bar
        self.progress_bar = ProgressBarWithText(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setText('Waiting...')

        # Grid layout for the widgets
        main_layout.addWidget(QLabel('Simulation Mode'), 0, 3)
        main_layout.addLayout(mode_layout, 1, 3)
        main_layout.addWidget(self.target_combo_box, 2, 3)
        main_layout.addLayout(mesh_param_layout, 3, 3)
        main_layout.addWidget(self.sdr_file, 4, 3)
        main_layout.addLayout(sdr_param_layout, 5, 3)
        main_layout.addWidget(self.target_save_path, 6, 3)
        main_layout.addWidget(self.clutter_save_path, 7, 3)
        main_layout.addLayout(rparam_layout, 8, 3)
        main_layout.addWidget(self.progress_bar, 16, 3)
        main_layout.addWidget(self.simulate_button, 17, 3)
        main_layout.addWidget(QLabel('Simulation Parameters'), 16, 0)
        main_layout.addLayout(position_layout, 17, 0)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # refresh speed in milliseconds
        timer.timeout.connect(self.openGL.updateGL)
        timer.start()

        self.loadPersistentSettings()

    def loadMesh(self):
        tinfo = self.target_info.loc[self.target_combo_box.currentIndex()]
        self._mesh_path = f"/home/jeff/Documents/target_meshes/{tinfo['filename']}"
        try:
            mesh = readCombineMeshFile(self._mesh_path, self.triangle_spinbox.value(), scale=1 / self.scaling_spinbox.value())
            self.openGL.set_mesh(mesh)
            self.updateGrid()
        except:
            pass

    def loadSAR(self, sdr_path):
        try:
            sdr = load(sdr_path)
            self._bg = SDREnvironment(sdr)
            self._rp = SDRPlatform(sdr, self._bg.ref)
            self.loadElevationMap()
            self.updateGrid()
        except:
            print('SAR not loaded.')

    def loadElevationMap(self):
        gx, gy, gz = self._bg.getGrid(self._bg.ref, self.grid_width.value(), self.grid_height.value(),
                                self.grid_nrows.value(), self.grid_ncols.value())
        self.updatePosition(gx.mean(), gy.mean(), gz.mean())
        self.elevation_map = np.dstack([gx.flatten(), gy.flatten(), gz.flatten()])


    def setMode(self):
        self._mode = self.mode_buttons.checkedButton().text()
        if self._mode == 'Profile':
            self.openGL.grid_mode = 0
            self.grid_ncols.setEnabled(False)
            self.grid_nrows.setEnabled(False)
            self.grid_width.setEnabled(False)
            self.grid_height.setEnabled(False)
            self.sdr_file.setEnabled(False)
            self.x_pos_spinbox.setEnabled(False)
            self.y_pos_spinbox.setEnabled(False)
            self.z_pos_spinbox.setEnabled(False)
            self.azimuth_spinbox.setEnabled(True)
            self.elevation_spinbox.setEnabled(True)
            self.range_spinbox.setEnabled(True)
            self.clutter_save_path.setEnabled(False)
            self.target_save_path.setEnabled(True)
        else:
            self.openGL.grid_mode = 1
            self.grid_ncols.setEnabled(True)
            self.grid_nrows.setEnabled(True)
            self.grid_width.setEnabled(True)
            self.grid_height.setEnabled(True)
            self.sdr_file.setEnabled(True)
            self.x_pos_spinbox.setEnabled(True)
            self.y_pos_spinbox.setEnabled(True)
            self.z_pos_spinbox.setEnabled(True)
            self.azimuth_spinbox.setEnabled(False)
            self.elevation_spinbox.setEnabled(False)
            self.range_spinbox.setEnabled(False)
            self.clutter_save_path.setEnabled(True)
            self.target_save_path.setEnabled(False)
        self.updateGrid()

    def updateGrid(self):
        if self._mode == 'Profile':
            self.openGL.update_grid_param(self.range_spinbox.value(), az_samples=self.azimuth_spinbox.value(),
                                          el_samples=self.elevation_spinbox.value())
        else:
            self.openGL.update_grid_param(el_map=self.elevation_map - self.virtual_pos)

    def updatePosition(self, x, y, z):
        self.virtual_pos = np.array([x, y, z])
        self.x_pos_spinbox.setValue(x)
        self.y_pos_spinbox.setValue(y)
        self.z_pos_spinbox.setValue(z)

    def setPosition(self):
        self.updatePosition(self.x_pos_spinbox.value(), self.y_pos_spinbox.value(), self.z_pos_spinbox.value())
        self.updateGrid()


    def loadPersistentSettings(self):
        settings = QSettings("ARTEMIS_SIM", "Simulator")
        self.triangle_spinbox.setValue(int(settings.value("mesh_ntris", 10000)))
        self.scaling_spinbox.setValue(float(settings.value("mesh_scaling", 7.)))
        self.grid_ncols.setValue(int(settings.value("grid_ncols", 10)))
        self.grid_nrows.setValue(int(settings.value("grid_nrows", 10)))
        self.grid_height.setValue(float(settings.value("grid_height", 10.)))
        self.grid_width.setValue(float(settings.value("grid_width", 10.)))
        self.target_combo_box.setCurrentIndex(int(settings.value("current_target", 0)))
        self.azimuth_spinbox.setValue(int(settings.value("az_samples", 32)))
        self.elevation_spinbox.setValue(int(settings.value("el_samples", 32)))
        self.range_spinbox.setValue(float(settings.value("range", 500.)))
        self.sdr_file.line_edit.setText(settings.value("sar_file", ""))
        self.target_save_path.line_edit.setText(settings.value('target_save_path', ""))
        self.clutter_save_path.line_edit.setText(settings.value('clutter_save_path', ""))
        self.iteration_spinbox.setValue(settings.value('n_iters', 5))
        self.fc_spinbox.setValue(settings.value('fc', 5))
        self.rx_gain_spinbox.setValue(settings.value('rx_gain', 5))
        self.tx_gain_spinbox.setValue(settings.value('tx_gain', 5))
        self.tx_power_spinbox.setValue(settings.value('tx_power', 5))
        self.rec_gain_spinbox.setValue(settings.value('rec_gain', 5))
        mode = settings.value("sim_mode", '')
        if mode == 'Profile':
            self.mode_target.setChecked(True)
            self.loadMesh()
        elif mode == 'SDR Train':
            self.mode_train.setChecked(True)
            self.loadSAR(self.sdr_file.line_edit.text())

    def savePersistentSettings(self):
        settings = QSettings("ARTEMIS_SIM", "Simulator")
        settings.setValue('mesh_ntris', self.triangle_spinbox.value())
        settings.setValue('mesh_scaling', self.scaling_spinbox.value())
        settings.setValue('grid_ncols', self.grid_ncols.value())
        settings.setValue('grid_nrows', self.grid_nrows.value())
        settings.setValue('grid_width', self.grid_width.value())
        settings.setValue('grid_height', self.grid_height.value())
        settings.setValue('az_samples', self.azimuth_spinbox.value())
        settings.setValue('el_samples', self.elevation_spinbox.value())
        settings.setValue('range', self.range_spinbox.value())
        settings.setValue('sar_file', self.sdr_file.line_edit.text())
        settings.setValue('current_target', self.target_combo_box.currentIndex())
        settings.setValue('target_save_path', self.target_save_path.line_edit.text())
        settings.setValue('clutter_save_path', self.clutter_save_path.line_edit.text())
        settings.setValue('n_iters', self.iteration_spinbox.value())
        settings.setValue('fc', self.fc_spinbox.value())
        settings.setValue('tx_gain', self.tx_gain_spinbox.value())
        settings.setValue('rx_gain', self.rx_gain_spinbox.value())
        settings.setValue('rec_gain', self.rec_gain_spinbox.value())
        settings.setValue('tx_power', self.tx_power_spinbox.value())
        settings.setValue('sim_mode', self._mode)

    def closeEvent(self, a_event, **kwargs):
        self.savePersistentSettings()
        a_event.accept()

    def run_simulation(self):
        mesh = readCombineMeshFile(self._mesh_path, self.triangle_spinbox.value(),
                                   scale=1 / self.scaling_spinbox.value())
        self.thread = SimulationThread(self._mode, mesh, self.sdr_file.line_edit.text(), self.clutter_save_path.line_edit.text(),
                                       self.target_save_path.line_edit.text(), True, self.scaling_spinbox.value(), self.target_combo_box.currentIndex(),
                                       TruncatedSVD(n_components=self.azimuth_spinbox.value() * self.elevation_spinbox.value()),
                                       self.azimuth_spinbox.value(), self.elevation_spinbox.value(), self.radar_coeff,
                                       n_iters=self.iteration_spinbox.value(), fft_len=8192)
        self.thread.signal_update_progress.connect(self.slot_update_progress)
        self.thread.signal_update_percentage.connect(self.slot_update_percentage)
        # self.thread.signal_waveform_generated.connect(self.slot_updatePlot)
        # self.thread.finished.connect(lambda: self.toggleGUIElements(True))
        self.thread.start()

    def slot_update_progress(self, value):
        self.progress_bar.setText(value)

    def slot_update_percentage(self, value):
        self.progress_bar.setValue(value)

    def slot_set_radar_coeff(self):
        self.radar_coeff = get_radar_coeff(self.fc_spinbox.value(), self.tx_power_spinbox.value(),
                                           self.rx_gain_spinbox.value(), self.tx_gain_spinbox.value(),
                                           self.rec_gain_spinbox.value())


class SimulationThread(QThread):
    signal_update_progress = pyqtSignal(str)
    signal_update_percentage = pyqtSignal(int)
    signal_waveform_generated = pyqtSignal(object)

    def __init__(self, mode, mesh, clutter_file_path, tensor_clutter_path, tensor_target_path, save_files, scaling,
                 target_index, tsvd, n_az_samples, n_el_samples, radar_coeff, n_iters=5, fft_len=8192):
        super().__init__()
        self.mesh = mesh
        self.clut = clutter_file_path
        self.tensor_clutter_path = tensor_clutter_path
        self.tensor_target_path = tensor_target_path
        self.save_files = save_files
        self.mode = 0 if mode == 'Profile' else 1
        self.scaling = scaling
        self.tidx = target_index
        self.tsvd = tsvd
        self.fft_len = fft_len
        self.n_az_samples = n_az_samples
        self.n_el_samples = n_el_samples
        self.radar_coeff = radar_coeff
        self.n_iters = n_iters

    def run(self):
        if self.mode == 1:
            abs_clutter_idx = 0
            self.signal_update_progress.emit('Loading clutter and target spectra...')
            for ntpsd, sdata in loadClutterTargetSpectrum(self.clut, self.radar_coeff, self.mesh, self.scaling):
                self.signal_update_progress.emit('Running simulation...')
                if self.save_files:
                    for nt, sd in zip(ntpsd, sdata):
                        if not np.any(np.isnan(nt)) and not np.any(np.isnan(sd)):
                            torch.save([torch.tensor(sd, dtype=torch.float32),
                                        torch.tensor(nt, dtype=torch.float32), self.tidx],
                                       f"{self.tensor_clutter_path}/tc_{abs_clutter_idx}.pt")
                            abs_clutter_idx += 1
                            self.signal_update_percentage.emit(abs_clutter_idx * 20)
        elif self.mode == 0:
            mf_chirp = np.fft.fft(genChirp(4300, 2e9, 9.6e9, 400e6), self.fft_len)
            mf_chirp = mf_chirp * mf_chirp.conj()
            streams = [cuda.stream() for _ in range(1)]
            self.signal_update_progress.emit('Loading mesh parameters...')
            gen_iter = iter(genProfileFromMesh('', self.n_iters, mf_chirp, 4,
                                               2**16, self.scaling, streams, [500., 25000.], self.fft_len,
                                               self.radar_coeff, 4600, 2e9, 9.6e9, 1, a_mesh=self.mesh,
                                               a_naz=self.n_az_samples, a_nel=self.n_el_samples))
            self.signal_update_progress.emit('Running simulation...')
            abs_idx = 0
            for rprof, pd, i in gen_iter:
                if np.all(pd == 0):
                    print(f'Skipping on target {self.tidx}, pd {i}')
                    continue

                pd_cat = processTargetProfile(pd, self.fft_len, self.tsvd)

                # Append to master target list
                if self.save_files and pd_cat is not None:
                    # Save the block out to a torch file for the dataloader later
                    torch.save([torch.tensor(pd_cat, dtype=torch.float32), self.tidx],
                               f"{self.tensor_target_path}/target_{self.tidx}_{abs_idx}.pt")
                    abs_idx += 1
                self.signal_update_percentage.emit(abs_idx * 20)
        self.signal_update_progress.emit('Done simulating.')



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    sys.exit(app.exec_())
