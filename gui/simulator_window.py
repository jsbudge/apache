from functools import partial
import moderngl
import pandas as pd
import torch
from PyQt5.QtCore import QThread, QSettings, pyqtSignal, Qt, QTimer
from OpenGL.GLUT import *
from PyQt5.QtWidgets import QAction, QHBoxLayout, QVBoxLayout, QGridLayout, QWidget, QDoubleSpinBox, QLabel, \
    QRadioButton, QButtonGroup, QPushButton, QComboBox, QSplashScreen, QApplication, QMainWindow
from sdrparse.SDRParsing import SDRBase, load
from simulib.grid_helper import SDREnvironment
from simulib.mesh_functions import readCombineMeshFile
from simulib.platform_helper import SDRPlatform
from simulib.simulation_functions import genChirp, llh2enu, enu2llh
from sklearn.decomposition import TruncatedSVD
from simulib.backproject_functions import getRadarAndEnvironment
from superqt import QLargeIntSpinBox
from utils import get_radar_coeff
from numba import cuda
from gui.gui_classes import FileSelectWidget, ProgressBarWithText
from mesh_viewer import QGLControllerWidget, ball
import gui_utils as f
import argparse
import numpy as np
from target_data_generator import loadClutterTargetSpectrum, getTargetProfile, processTargetProfile, genProfileFromMesh

DTR = np.pi / 180.


class MainWindow(QMainWindow):
    patterns: list
    thread: QThread = None
    _rp: SDRPlatform = None
    _bg: SDREnvironment = None
    elevation_map: np.array = None
    platform_path: np.array = None
    virtual_pos: np.array = np.array([0., 0., 0.])  # Lat, Lon, Alt
    virtual_att: np.array = np.array([0., 0., 0.])
    grid_vals: tuple[float, float, int, int] = (10., 10., 10, 10)
    radar_params: tuple[float, float, float, float, float] = (7, 32, 32, 100, 100)
    profile_params: tuple[int, int, float] = (8, 8, 1200.)
    mesh_params: tuple[float, int, int] = (7., 10000, 5)
    target_params: tuple[int, str, str, str]
    to_update: np.array = None
    radar_coeff: float = 0.
    win_width: int = 500
    win_height: int = 500
    win_full_width: int = 1200
    _model_fnme: str = None
    _mesh_path: str = None
    _mode: str = 'Profile'

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simulator")
        self.setGeometry(200, 200, self.win_width, self.win_height)
        # self.setFixedSize(self.win_width, self.win_height)
        # menu_bar = self.menuBar()
        # new_window_action = QAction("Settings", self)
        # new_window_action.triggered.connect(self.open_settings_window)
        # menu_bar.addAction(new_window_action)

        self.target_info = pd.read_csv('../data/target_info.csv')
        self.loadPersistentSettings()

        main_layout = QGridLayout()

        self.target_combo_box = QComboBox(self)
        self.target_combo_box.addItems(self.target_info['name'])
        self.target_combo_box.setCurrentIndex(self.target_params[0])
        # self.target_combo_box.lineEdit().setReadOnly(True)
        self.target_combo_box.currentIndexChanged.connect(self.slot_update_mesh)

        self.sdr_file = FileSelectWidget(self, 'Select SDR File', file_types="SAR Files (*.sar)")
        self.sdr_file.line_edit.setText(self.target_params[1])
        self.sdr_file.signal_btn_clicked.connect(self.slot_load_sar)

        position_layout = QGridLayout()

        self.lat_spinbox = QDoubleSpinBox(self)
        self.lat_spinbox.setRange(-90., 90.)
        self.lat_spinbox.setDecimals(10)
        self.lat_spinbox.setFixedWidth(120)
        self.lat_spinbox.setValue(self.virtual_pos[0])
        self.lon_spinbox = QDoubleSpinBox(self)
        self.lon_spinbox.setRange(-180., 180.)
        self.lon_spinbox.setDecimals(10)
        self.lon_spinbox.setFixedWidth(120)
        self.lon_spinbox.setValue(self.virtual_pos[1])
        self.alt_spinbox = QDoubleSpinBox(self)
        self.alt_spinbox.setRange(-1000., 10000.)
        self.alt_spinbox.setDecimals(10)
        self.alt_spinbox.setFixedWidth(120)
        self.alt_spinbox.setValue(self.virtual_pos[2])
        self.lat_spinbox.valueChanged.connect(self.slot_update_position)
        self.lon_spinbox.valueChanged.connect(self.slot_update_position)
        self.alt_spinbox.valueChanged.connect(self.slot_update_position)
        self.x_att_spinbox = QDoubleSpinBox(self)
        self.x_att_spinbox.setRange(-360., 360.)
        self.x_att_spinbox.setFixedWidth(70)
        self.y_att_spinbox = QDoubleSpinBox(self)
        self.y_att_spinbox.setRange(-360., 360.)
        self.y_att_spinbox.setFixedWidth(70)
        self.z_att_spinbox = QDoubleSpinBox(self)
        self.z_att_spinbox.setRange(-360., 360.)
        self.z_att_spinbox.setFixedWidth(70)
        self.x_att_spinbox.valueChanged.connect(self.slot_update_attitude)
        self.y_att_spinbox.valueChanged.connect(self.slot_update_attitude)
        self.z_att_spinbox.valueChanged.connect(self.slot_update_attitude)
        self.azimuth_spinbox = QLargeIntSpinBox(self)
        self.azimuth_spinbox.setValue(self.profile_params[0])
        self.elevation_spinbox = QLargeIntSpinBox(self)
        self.elevation_spinbox.setValue(self.profile_params[1])
        self.range_spinbox = QDoubleSpinBox(self)
        self.range_spinbox.setValue(self.profile_params[2])
        self.range_spinbox.setRange(1., 25000.)
        self.azimuth_spinbox.valueChanged.connect(self.slot_update_profile_background)
        self.elevation_spinbox.valueChanged.connect(self.slot_update_profile_background)
        self.range_spinbox.valueChanged.connect(self.slot_update_profile_background)
        position_layout.addWidget(self.lat_spinbox, 0, 1)
        position_layout.addWidget(self.lon_spinbox, 0, 3)
        position_layout.addWidget(self.alt_spinbox, 0, 5)
        position_layout.addWidget(self.x_att_spinbox, 1, 1)
        position_layout.addWidget(self.y_att_spinbox, 1, 3)
        position_layout.addWidget(self.z_att_spinbox, 1, 5)
        position_layout.addWidget(self.azimuth_spinbox, 2, 1)
        position_layout.addWidget(self.elevation_spinbox, 2, 3)
        position_layout.addWidget(self.range_spinbox, 2, 5)
        position_layout.addWidget(QLabel('Lat:'), 0, 0)
        position_layout.addWidget(QLabel('Lon:'), 0, 2)
        position_layout.addWidget(QLabel('Alt:'), 0, 4)
        position_layout.addWidget(QLabel('R:'), 1, 0)
        position_layout.addWidget(QLabel('P:'), 1, 2)
        position_layout.addWidget(QLabel('Y:'), 1, 4)
        position_layout.addWidget(QLabel('# Az:'), 2, 0)
        position_layout.addWidget(QLabel('# El:'), 2, 2)
        position_layout.addWidget(QLabel('Range:'), 2, 4)

        self.mode_target = QRadioButton('Profile')
        self.mode_train = QRadioButton('SDR Train')
        self.mode_buttons = QButtonGroup()
        self.mode_buttons.addButton(self.mode_target)
        self.mode_buttons.addButton(self.mode_train)
        if self._mode == 'Profile':
            self.mode_target.setChecked(True)
        else:
            self.mode_train.setChecked(True)
        self.mode_buttons.buttonToggled.connect(self.setMode)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.mode_target)
        mode_layout.addWidget(self.mode_train)

        mesh_param_layout = QHBoxLayout()
        self.triangle_spinbox = QLargeIntSpinBox()
        self.triangle_spinbox.setRange(10, 5e6)
        self.triangle_spinbox.setValue(self.mesh_params[1])
        self.triangle_spinbox.valueChanged.connect(self.slot_update_mesh)
        self.scaling_spinbox = QDoubleSpinBox()
        self.scaling_spinbox.setRange(.01, 1000)
        self.scaling_spinbox.setValue(self.mesh_params[0])
        self.scaling_spinbox.valueChanged.connect(self.slot_update_mesh)
        mesh_param_layout.addWidget(QLabel('# Tris:'))
        mesh_param_layout.addWidget(self.triangle_spinbox)
        mesh_param_layout.addWidget(QLabel('Scaling Factor:'))
        mesh_param_layout.addWidget(self.scaling_spinbox)

        sdr_param_layout = QHBoxLayout()
        self.grid_width = QDoubleSpinBox()
        self.grid_width.setRange(1., 1000.)
        self.grid_width.setValue(self.grid_vals[0])
        self.grid_height = QDoubleSpinBox()
        self.grid_height.setRange(1., 1000.)
        self.grid_height.setValue(self.grid_vals[1])
        self.grid_nrows = QLargeIntSpinBox()
        self.grid_nrows.setRange(10, 100)
        self.grid_nrows.setValue(self.grid_vals[2])
        self.grid_ncols = QLargeIntSpinBox()
        self.grid_ncols.setRange(10, 100)
        self.grid_ncols.setValue(self.grid_vals[3])
        sdr_param_layout.addWidget(QLabel('Width:'))
        sdr_param_layout.addWidget(self.grid_width)
        sdr_param_layout.addWidget(QLabel('Height:'))
        sdr_param_layout.addWidget(self.grid_height)
        sdr_param_layout.addWidget(QLabel('Rows:'))
        sdr_param_layout.addWidget(self.grid_nrows)
        sdr_param_layout.addWidget(QLabel('Cols:'))
        sdr_param_layout.addWidget(self.grid_ncols)
        self.reload_button = QPushButton('Reload Grid')
        self.reload_button.clicked.connect(self.slot_reload_background)

        # File saving
        self.target_save_path = FileSelectWidget(self, 'Target Profile Save Path', read_only=False)
        self.target_save_path.line_edit.setText(self.target_params[2])
        self.clutter_save_path = FileSelectWidget(self, 'SAR Train Save Path', read_only=False)
        self.clutter_save_path.line_edit.setText(self.target_params[3])

        # Radar parameters
        rparam_layout = QHBoxLayout()
        rparam_sec_layout = QHBoxLayout()
        self.fc_spinbox = QDoubleSpinBox()
        self.fc_spinbox.setValue(self.radar_params[0])
        self.fc_spinbox.setRange(8000, 10000)
        rparam_layout.addWidget(QLabel('FC (MHz):'))
        rparam_layout.addWidget(self.fc_spinbox)
        self.tx_power_spinbox = QDoubleSpinBox()
        self.tx_power_spinbox.setValue(self.radar_params[1])
        self.tx_power_spinbox.setRange(10, 100)
        rparam_layout.addWidget(QLabel('Tx Power (dB):'))
        rparam_layout.addWidget(self.tx_power_spinbox)
        self.tx_gain_spinbox = QDoubleSpinBox()
        self.tx_gain_spinbox.setValue(self.radar_params[2])
        self.tx_gain_spinbox.setRange(1, 500)
        rparam_sec_layout.addWidget(QLabel('Tx Gain (dB):'))
        rparam_sec_layout.addWidget(self.tx_gain_spinbox)
        self.rx_gain_spinbox = QDoubleSpinBox()
        self.rx_gain_spinbox.setValue(self.radar_params[3])
        self.rx_gain_spinbox.setRange(1, 500)
        rparam_sec_layout.addWidget(QLabel('Rx Gain (dB):'))
        rparam_sec_layout.addWidget(self.rx_gain_spinbox)
        self.rec_gain_spinbox = QDoubleSpinBox()
        self.rec_gain_spinbox.setValue(self.radar_params[4])
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
        self.iteration_spinbox.setValue(self.mesh_params[2])
        self.iteration_spinbox.setRange(1, 100)
        rparam_sec_layout.addWidget(QLabel('# Files to Generate:'))
        rparam_sec_layout.addWidget(self.iteration_spinbox)

        # Progress Bar
        self.progress_bar = ProgressBarWithText(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setText('Waiting...')

        # Create openGL context
        self.openGL = QGLControllerWidget(self, 0 if self._mode == 'Profile' else 1)
        self.openGL.setFixedSize(500, 500)
        main_layout.addWidget(self.openGL, 0, 0, 15, 3)
        self.simulate_button = QPushButton('Simulate!')
        self.simulate_button.clicked.connect(self.run_simulation)

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
        main_layout.addLayout(rparam_sec_layout, 9, 3)
        main_layout.addWidget(self.reload_button, 10, 3)
        main_layout.addWidget(self.progress_bar, 16, 3)
        main_layout.addWidget(self.simulate_button, 17, 3)
        main_layout.addWidget(QLabel('Simulation Parameters'), 16, 0)
        main_layout.addLayout(position_layout, 17, 0)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        if self._mode == 'Profile':
            self._setmode(0, False, True)
        else:
            self._setmode(1, True, False)

        timer = QTimer(self)
        timer.setInterval(20)  # refresh speed in milliseconds
        timer.timeout.connect(self.openGL.updateGL)
        timer.start()

        init_timer = QTimer(self)
        init_timer.setSingleShot(True)
        init_timer.setInterval(1000)
        init_timer.timeout.connect(self.initialize)
        init_timer.start()



    def setMode(self):
        self._mode = self.mode_buttons.checkedButton().text()
        if self._mode == 'Profile':
            self._setmode(0, False, True)
            self.slot_update_profile_background()
        else:
            self._setmode(1, True, False)
            self.slot_reload_background()

    def _setmode(self, arg0, arg1, arg2):
        self.openGL.grid_mode = arg0
        self.grid_ncols.setEnabled(arg1)
        self.grid_nrows.setEnabled(arg1)
        self.grid_width.setEnabled(arg1)
        self.grid_height.setEnabled(arg1)
        self.sdr_file.setEnabled(arg1)
        self.lat_spinbox.setEnabled(arg1)
        self.lon_spinbox.setEnabled(arg1)
        self.alt_spinbox.setEnabled(arg1)
        self.azimuth_spinbox.setEnabled(arg2)
        self.elevation_spinbox.setEnabled(arg2)
        self.range_spinbox.setEnabled(arg2)
        self.clutter_save_path.setEnabled(arg1)
        self.target_save_path.setEnabled(arg2)


    def initialize(self):
        self.slot_update_mesh()
        self.setMode()


    def loadPersistentSettings(self):
        settings = QSettings("ARTEMIS_SIM", "Simulator")
        self.mesh_params = (float(settings.value("mesh_scaling", 7.)), int(settings.value("mesh_ntris", 10000)),
                            settings.value('n_iters', 5))
        self.grid_vals = (float(settings.value("grid_height", 10.)), float(settings.value("grid_width", 10.)),
                          int(settings.value("grid_ncols", 10)), int(settings.value("grid_nrows", 10)))
        self.target_params = (int(settings.value("current_target", 0)), settings.value("sar_file", ""),
                              settings.value('target_save_path', ""), settings.value('clutter_save_path', ""))
        self.profile_params = (int(settings.value("az_samples", 32)), int(settings.value("el_samples", 32)),
                               float(settings.value("range", 500.)))
        self.radar_params = (float(settings.value('fc', 5)), float(settings.value('rx_gain', 5)), float(settings.value('tx_gain', 5)),
                             float(settings.value('tx_power', 5)), float(settings.value('rec_gain', 5)))
        self.virtual_pos = np.array([float(settings.value('lat', 40.138044)), float(settings.value('lon', -111.660027)),
                                     float(settings.value('alt', 1365.8849123907273))])
        self._mode = settings.value("sim_mode", 'Profile')

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
        settings.setValue('lat', self.lat_spinbox.value())
        settings.setValue('lon', self.lon_spinbox.value())
        settings.setValue('alt', self.alt_spinbox.value())

    def closeEvent(self, a_event, **kwargs):
        self.savePersistentSettings()
        a_event.accept()

    def run_simulation(self):
        self.thread = SimulationThread(self._mode, self.openGL.mesh, self.sdr_file.line_edit.text(),
                                       self.clutter_save_path.line_edit.text(), self.target_save_path.line_edit.text(),
                                       True, self.scaling_spinbox.value(), self.target_combo_box.currentIndex(),
                                       TruncatedSVD(n_components=self.azimuth_spinbox.value() *
                                                                 self.elevation_spinbox.value()),
                                       self.azimuth_spinbox.value(), self.elevation_spinbox.value(), self.radar_coeff,
                                       n_iters=self.iteration_spinbox.value(), fft_len=8192)
        self.thread.signal_update_progress.connect(self.slot_update_progress)
        self.thread.signal_update_percentage.connect(self.slot_update_percentage)
        self.thread.start()

    def updateProgress(self, p=None, i=None):
        if i is not None:
            self.progress_bar.setText(i)
        if p is not None:
            self.progress_bar.setValue(int(p))

    def slot_update_progress(self, value):
        self.progress_bar.setText(value)

    def slot_update_percentage(self, value):
        self.progress_bar.setValue(value)

    def slot_update_position(self, centered=True):
        self.virtual_pos = np.array([self.lat_spinbox.value(), self.lon_spinbox.value(), self.alt_spinbox.value()])
        self.openGL.modify_mesh(pos=llh2enu(*self.virtual_pos, self._bg.ref if centered else self.virtual_pos))
        self.openGL.look_at(self.openGL.mesh.get_center(), self.openGL.mesh.get_center() + np.array([15., 0, 0]))
        self.updateProgress(i='Updated Position')

    def slot_update_attitude(self):
        att = np.array([self.x_att_spinbox.value(), self.y_att_spinbox.value(), self.z_att_spinbox.value()]) * DTR
        self.openGL.modify_mesh(att=self.virtual_att - att)
        self.virtual_att = att
        self.updateProgress(i='Updated Attitude')

    def slot_update_mesh(self):
        tinfo = self.target_info.loc[self.target_combo_box.currentIndex()]
        self.scaling_spinbox.setValue(tinfo['scaling'])
        self._mesh_path = f"/home/jeff/Documents/target_meshes/{tinfo['filename']}"
        self.updateProgress(i='Reading mesh file...')
        try:
            mesh = readCombineMeshFile(self._mesh_path, self.triangle_spinbox.value(),
                                       scale=1 / self.scaling_spinbox.value())
            self.updateProgress(70, 'Loading mesh...')
            self.openGL.set_mesh(mesh)
            self.updateProgress(0, 'Mesh updated.')
        except FileNotFoundError:
            print('Mesh file not found.')
        if self._mode == 'Profile':
            self.slot_update_profile_background()
        else:
            self.slot_reload_background()

    def slot_update_profile_background(self):
        self.openGL.vaos = []
        # if len(self.openGL.vaos) == 0:
        self.openGL.add_grid(
            ball(self.range_spinbox.value(), self.azimuth_spinbox.value(), self.elevation_spinbox.value(), False))
        self.slot_update_position(False)

        self.updateProgress(i='Antenna profile loaded.')

    def slot_reload_background(self):
        if self._bg is None:
            self.slot_load_sar()
        self.updateProgress(0, i='Generating elevation grid...')
        gx, gy, gz = self._bg.getGrid(self.virtual_pos, self.grid_width.value(), self.grid_height.value(),
                                      self.grid_nrows.value(), self.grid_ncols.value())
        self.updateProgress(70, i='Generating flight profile...')
        self.elevation_map = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T
        '''self.elevation_map = np.concatenate(
            (np.dstack([gx[:, :-1].flatten(), gy[:, :-1].flatten(), gz[:, :-1].flatten()]),
             np.dstack([gx[:, 1:].flatten(), gy[:, 1:].flatten(), gz[:, 1:].flatten()])))'''

        if len(self.openGL.vaos) == 0:
            self.openGL.add_grid(self.elevation_map)
        else:
            self.openGL.update_grid(0, self.elevation_map)
        if len(self.openGL.vaos) == 1:
            self.openGL.add_grid(self.platform_path)
        else:
            self.openGL.update_grid(1, self.platform_path)
        self.updateProgress(90, i='Centering on model...')
        self.slot_update_position()
        self.openGL.look_at(llh2enu(*self.virtual_pos, self._bg.ref), self.platform_path[0])
        # self.openGL.fov = 1000.
        self.updateProgress(0, 'Background updated.')

    def slot_load_sar(self):
        self.updateProgress(0, i='Loading SDR file...')
        sdr = load(self.sdr_file.line_edit.text())
        self.updateProgress(35, 'Generating environment...')
        self._bg, self._rp = getRadarAndEnvironment(sdr)
        self.updateProgress(80, 'Loading platform...')
        self.platform_path = self._rp.pos(self._rp.gpst)
        self.updateProgress(0, 'SDR file loaded.')

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
    app = QApplication(sys.argv)
    '''spl_pix = QPixmap('../artemislogo.png')
    splash = QSplashScreen(spl_pix, Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage('Loading wavemodel...')'''
    win = MainWindow()
    # splash.close()
    win.show()

    sys.exit(app.exec_())
