from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QIcon
from OpenGL.GLUT import *
from PyQt5.QtWidgets import QAction, QHBoxLayout, QVBoxLayout, QGridLayout, QWidget, QDoubleSpinBox, QLabel
from sdrparse.SDRParsing import SDRBase
from superqt import QLargeIntSpinBox

from gui.gui_classes import FileSelectWidget
from mesh_viewer import QGLControllerWidget
import gui_utils as f
import argparse


class MainWindow(QtWidgets.QMainWindow):
    patterns: list
    thread: QThread = None
    sdr: SDRBase = None
    win_width: int = 500
    win_height: int = 500
    win_full_width: int = 1200
    _model_fnme: str = None
    _target_names_file: str = '../target_files.yaml'
    _target_ids_file: str = '../data/target_ids.txt'
    _target_mesh_path: str = '/home/jeff/Documents/target_meshes'
    _model_path: str = '../vae_config.yaml'

    def __init__(self):
        # Load UI file
        QtWidgets.QMainWindow.__init__(self)

        self.setWindowTitle("Waveform Generator")
        self.setGeometry(200, 200, self.win_width, self.win_height)
        self.setFixedSize(self.win_width, self.win_height)
        menu_bar = self.menuBar()
        new_window_action = QAction("Settings", self)
        new_window_action.triggered.connect(self.open_settings_window)
        menu_bar.addAction(new_window_action)

        # Create openGL context
        self.openGL = QGLControllerWidget(self)
        self.openGL.setGeometry(0, 37, 870, 731)
        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # refresh speed in milliseconds
        timer.timeout.connect(self.openGL.updateGL)
        timer.start()

        self.mesh_file = FileSelectWidget(self, 'Select Mesh')
        self.sdr_file = FileSelectWidget(self, 'Select SDR File')

        position_layout = QGridLayout()

        self.x_pos_spinbox = QDoubleSpinBox(self)
        self.y_pos_spinbox = QDoubleSpinBox(self)
        self.z_pos_spinbox = QDoubleSpinBox(self)
        self.azimuth_spinbox = QLargeIntSpinBox(self)
        self.elevation_spinbox = QLargeIntSpinBox(self)
        position_layout.addWidget(self.x_pos_spinbox, 0, 1)
        position_layout.addWidget(self.y_pos_spinbox, 0, 3)
        position_layout.addWidget(self.z_pos_spinbox, 0, 5)
        position_layout.addWidget(self.azimuth_spinbox, 1, 1)
        position_layout.addWidget(self.elevation_spinbox, 1, 3)
        position_layout.addWidget(QLabel('X:'), 0, 0)
        position_layout.addWidget(QLabel('Y:'), 0, 2)
        position_layout.addWidget(QLabel('Z:'), 0, 4)
        position_layout.addWidget(QLabel('# Az:'), 1, 0)
        position_layout.addWidget(QLabel('# El:'), 1, 2)
        # Main layout
        main_layout = QHBoxLayout()

        # Grid layout for the widgets
        grid_layout = QGridLayout()
        main_layout.addLayout(grid_layout)

        main_layout.addWidget(self.openGL, 0, 0, 3, 3)
        main_layout.addWidget(self.mesh_file, 0, 4)
        main_layout.addWidget(self.sdr_file, 1, 4)
        main_layout.addLayout(position_layout, 2, 4)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    sys.exit(app.exec_())
