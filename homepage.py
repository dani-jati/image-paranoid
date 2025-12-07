import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout,
    QPushButton, QLabel, QVBoxLayout, QSizePolicy
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import QSize, Qt

# Import your tools
from tools.cropper import Dashboard as CropperTool

from tools.protonose_altitude import Dashboard as ProtonoseAltitudeTool
from tools.eye_inclination import Dashboard as EyeInclinationTool
from tools.eye_zoom import Dashboard as EyeZoomTool
from tools.face_width import Dashboard as FaceWidthTool
from tools.forehead_height import Dashboard as ForeheadHeightTool
from tools.head_height import Dashboard as HeadHeightTool
from tools.head_size import Dashboard as HeadSizeTool
from tools.interocular_width import Dashboard as interocularWidthTool
from tools.midface_height import Dashboard as MidFaceHeightTool
from tools.mouth_width import Dashboard as MouthWidthTool
from tools.neck_width import Dashboard as NeckWidthTool
from tools.shoulder_tilt import Dashboard as ShoulderTiltTool

class HomePage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Paranoid — Home Screen")
        self.resize(800, 600)

        # Central widget with wallpaper background
        central = QWidget()

        self.setCentralWidget(central)
        central.setStyleSheet("""
            QWidget {
                border-image: url(icons/wallpaper.jpg) 0 0 0 0 stretch stretch;
            }
        """)

        grid = QGridLayout(central)
        grid.setSpacing(40)

        # Define apps: (icon_path, label, callback)
        apps = [
            ("icons/cropper.png", "Cropper", self.run_cropper_script),
            ("icons/protonose-altitude.png", "Proto-nose Altitude", self.open_eye_altitude),
            ("icons/eye-inclination.png", "Eye Inclination", self.open_eye_inclination),
            ("icons/eye-zoom.png", "Eye Zoom", self.run_eye_zoom),
            ("icons/face-width.png", "Face Width", self.open_face_width),
            ("icons/forehead-height.png", "Forehead Height", self.open_forehead_height),
            ("icons/head-height.png", "Head Height", self.open_head_height),
            ("icons/head-size.png", "Head Size", self.open_head_size),
            ("icons/interocular-width.png", "interocular Width", self.open_interocular_width),
            ("icons/midface-height.png", "Mid-Face Height", self.open_midface_height),
            ("icons/mouth-width.png", "Mouth Width", self.open_mouth_width),
            ("icons/neck-width.png", "Neck Width", self.open_neck_width),
            ("icons/shoulder-tilt.png", "Shoulder Tilt", self.open_shoulder_tilt),
        ]

        for i, (icon_path, name, callback) in enumerate(apps):
            # Create icon button
            btn = QPushButton()
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(64, 64))
            btn.setStyleSheet("text-align: center; margin-top: 0px;")
            btn.setFlat(True)
            btn.clicked.connect(callback)

            # Label styled for visibility on wallpaper
            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: white; font-weight: bold; font-size: 12px; margin:0px; padding:0px;")

            # Wrap button + label in a vertical layout
            vbox = QVBoxLayout()
            vbox.setSpacing(1)
            vbox.setContentsMargins(0,0,0,0)
 
            vbox.addWidget(btn, alignment=Qt.AlignCenter)
            vbox.addWidget(label, alignment=Qt.AlignCenter)

            vbox_widget = QWidget()
            vbox_widget.setLayout(vbox)
            vbox_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            grid.addWidget(vbox_widget, i // 3, i % 3, alignment=Qt.AlignCenter)

            # Place in grid (2 columns)
            # grid.addLayout(vbox, i // 2, i % 2)

    def open_shoulder_tilt(self):
        self.tool = ShoulderTiltTool()
        self.tool.show()

    def open_head_height(self):
        self.tool = HeadHeightTool()
        self.tool.show()

    def open_face_width(self):
        self.tool = FaceWidthTool()
        self.tool.show()

    def open_midface_height(self):
        self.tool = MidFaceHeightTool()
        self.tool.show()

    def open_forehead_height(self):
        self.tool = ForeheadHeightTool()
        self.tool.show()

    def open_neck_width(self):
        self.tool = NeckWidthTool()
        self.tool.show()

    def open_head_size(self):
        self.tool = HeadSizeTool()
        self.tool.show()

    def open_interocular_width(self):
        self.tool = interocularWidthTool()
        self.tool.show()

    def open_mouth_width(self):
        self.tool = MouthWidthTool()
        self.tool.show()

    def open_eye_altitude(self):
        self.tool = ProtonoseAltitudeTool()
        self.tool.show()

    def open_eye_inclination(self):
        self.tool = EyeInclinationTool()
        self.tool.show()

    def run_eye_zoom(self):
        self.tool = EyeZoomTool()
        self.tool.process()

    def run_cropper_script(self):
        self.tool = CropperTool()
        self.tool.process()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    home = HomePage()
    home.show()
    sys.exit(app.exec())

