from functools import partial
import moderngl
import pandas as pd
import torch
from PyQt5.QtCore import QThread, QSettings, pyqtSignal, Qt, QTimer
from OpenGL.GLUT import *
from PyQt5.QtWidgets import QAction, QHBoxLayout, QVBoxLayout, QGridLayout, QWidget, QDoubleSpinBox, QLabel, \
    QRadioButton, QButtonGroup, QPushButton, QComboBox, QSplashScreen, QApplication, QMainWindow, QTabWidget
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
from gui.gui_classes import FileSelectWidget, ProgressBarWithText, MplWidget
from mesh_viewer import QGLControllerWidget, ball
import gui_utils as f
import argparse
import numpy as np
from pathlib import Path
from target_data_generator import loadClutterTargetSpectrum, getTargetProfile, processTargetProfile, genProfileFromMesh


class SimulationDataWindow(QMainWindow):

    def __init__(self, files):
        super().__init__()
        self.setWindowTitle("Simulation Results")
        self.setGeometry(250, 250, 500, 200)
        layout = QVBoxLayout()

        self.files = files

        tnames = [Path(f).stem for f in files]
        self.tabs = QTabWidget()
        for t in tnames:
            self.tabs.addTab(QWidget(), t)

        self.tabs.tabBarClicked.connect(self.slot_tab_clicked)

        self.viewer = MplWidget(self)

        layout.addWidget(self.tabs)
        layout.addWidget(self.viewer)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def slot_tab_clicked(self, idx):
        dt = torch.load(self.files[idx])
        self.viewer.plot_dopp_map(dt[0][0])
