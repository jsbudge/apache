from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QGridLayout, QLineEdit, QPushButton, QFileDialog


class SettingsWindow(QMainWindow):
    signal_save: pyqtSignal

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setGeometry(250, 250, 300, 200)

        central_widget = QWidget()
        layout = QGridLayout(central_widget)
        layout.addWidget(QLabel("Settings will not be saved until Save button is clicked."), 0, 0, 1, 3)
        layout.addWidget(QLabel("Model file:"), 1, 0)
        self.model_line_edit = QLineEdit(self)
        self.model_line_edit.setAcceptDrops(True)
        self.model_line_edit.setReadOnly(True)
        layout.addWidget(self.model_line_edit, 1, 1)
        self.model_browse_btn = QPushButton("Browse", self)
        self.model_browse_btn.clicked.connect(self.browse_model_file)
        layout.addWidget(self.model_browse_btn, 1, 2)

        layout.addWidget(QLabel("Target name file:"), 2, 0)
        self.tname_line_edit = QLineEdit(self)
        self.tname_line_edit.setAcceptDrops(True)
        self.tname_line_edit.setReadOnly(True)
        layout.addWidget(self.tname_line_edit, 2, 1)
        self.tname_browse_btn = QPushButton("Browse", self)
        self.tname_browse_btn.clicked.connect(self.browse_tname_file)
        layout.addWidget(self.tname_browse_btn, 2, 2)

        layout.addWidget(QLabel("Target ID file:"), 3, 0)
        self.tid_line_edit = QLineEdit(self)
        self.tid_line_edit.setAcceptDrops(True)
        self.tid_line_edit.setReadOnly(True)
        layout.addWidget(self.tid_line_edit, 3, 1)
        self.tid_browse_btn = QPushButton("Browse", self)
        self.tid_browse_btn.clicked.connect(self.browse_tid_file)
        layout.addWidget(self.tid_browse_btn, 3, 2)

        layout.addWidget(QLabel("Target mesh filepath:"), 4, 0)
        self.mesh_line_edit = QLineEdit(self)
        self.mesh_line_edit.setAcceptDrops(True)
        self.mesh_line_edit.setReadOnly(True)
        layout.addWidget(self.mesh_line_edit, 4, 1)
        self.mesh_browse_btn = QPushButton("Browse", self)
        self.mesh_browse_btn.clicked.connect(self.browse_mesh_path)
        layout.addWidget(self.mesh_browse_btn, 4, 2)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(lambda: self.signal_save.emit())
        layout.addWidget(self.save_button, 5, 0)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.chkpt);;All Files (*)")
        if file_path:
            self.model_line_edit.setText(file_path)

    def browse_tname_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Target Name File", "",
                                                   "Name Files (*.yaml);;All Files (*)")
        if file_path:
            self.tname_line_edit.setText(file_path)

    def browse_tid_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Target ID File", "", "Txt Files (*.txt);;All Files (*)")
        if file_path:
            self.tid_line_edit.setText(file_path)

    def browse_mesh_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Mesh Path", "", "All Files (*)")
        if file_path:
            self.mesh_line_edit.setText(file_path)

    def drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def model_file_drop_event(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.model_line_edit.setText(file_path)
