# This script is to measure shoulder tilt,
# which is measured by comparing the tilt to a line parallel to nose as an anchor.

import sys, os, cv2, datetime, math
import numpy as np
import mediapipe as mp

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QSizePolicy, QComboBox, QFileDialog
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QImage, QTextCursor, QAction, QPainter, QColor, QFont
from PySide6.QtCore import QSettings, Qt    

# === Folders ===
script_dir = os.path.dirname(os.path.abspath(__file__))

output_folders = {
    "high": os.path.join(script_dir, "../images/output_images/shoulder_tilt/16_or_more"),
    "high_asymmetric": os.path.join(script_dir, "../images/output_images/shoulder_tilt/16_or_more_asymmetric"),
    "mid": os.path.join(script_dir, "../images/output_images/shoulder_tilt/12_to_16"),
    "mid_asymmetric": os.path.join(script_dir, "../images/output_images/shoulder_tilt/12_to_16_asymmetric"),
    "low": os.path.join(script_dir, "../images/output_images/shoulder_tilt/12_or_less"),
    "low_asymmetric": os.path.join(script_dir, "../images/output_images/shoulder_tilt/12_or_less_asymmetric"),  
    "asymmetric_x": os.path.join(script_dir, "../images/output_images/shoulder_tilt/asymmetric_x"),
}

for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Ensure symlink from cropper output to shoulder-tilt input
cropper_output = os.path.normpath(os.path.join(script_dir, "..", "images", "output_images", "cropper"))
shoulder_tilt_input = os.path.normpath(os.path.join(script_dir, "..", "images", "input_images", "shoulder_tilt"))
path = shoulder_tilt_input

if os.path.isdir(cropper_output) and not os.path.exists(shoulder_tilt_input):

    if os.name == 'posix': #linux or mac
        os.symlink(cropper_output, shoulder_tilt_input) #linux
    elif os.name == 'nt': #windows
        os.symlink(cropper_output, shoulder_tilt_input, target_is_directory=True) #windows

    print(f"🔗 Symlink created: {shoulder_tilt_input} → {cropper_output}")
elif os.path.islink(shoulder_tilt_input):
    print(f"✅ Symlink already exists: {shoulder_tilt_input} → {os.readlink(shoulder_tilt_input)}")
elif os.path.isdir(shoulder_tilt_input):
    print(f"⚠️ Destination exists as a real folder: {shoulder_tilt_input} — not creating symlink.")
else:

    if not os.path.islink(path):
        print(f"{path} is not a symbolic link.")
        os.remove(path)

        if os.name == 'posix': #linux or mac
            os.symlink(cropper_output, shoulder_tilt_input) #linux
        elif os.name == 'nt': #windows
            os.symlink(cropper_output, shoulder_tilt_input, target_is_directory=True) #windows

        print(f"🔗 Symlink created: {shoulder_tilt_input} → {cropper_output}")

print("Symlink points to:", os.readlink(shoulder_tilt_input))

# === Helpers ===
def cvimg_to_qpix(img):
    h, w, ch = img.shape
    bytes_per_line = ch * w
    # OpenCV uses BGR; we keep it as BGR888 for speed
    qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg)

def calculate_true_tilt(p1, p2):
    import math
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = abs(math.degrees(angle_rad))
    acute = abs(90 - angle_deg)
    return 90 - acute

def list_input_files():
    output_folder_high = output_folders["high"]
    output_folder_low = output_folders["low"]
    processed = set(os.listdir(output_folder_high)) | set(os.listdir(output_folder_low))
    files = [
        os.path.join(shoulder_tilt_input, f)
        for f in sorted(os.listdir(shoulder_tilt_input))
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in processed
    ]
    return files

def classify_perspective(p1, p2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    if dx < 30:  # shoulders too close, probably aside
        return "Side"

    ratio = dy / dx
    if ratio > 0.4:
        return "Diagonal"
    else:
        return "Front"


# === Clickable Label ===
class ClickableLabel(QLabel):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and hasattr(self, "parent_dash"):
            self.parent_dash.on_click(event)

# === Dashboard ===
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shoulder Tilt Dashboard")
        self.resize(1200, 800)

        # shoulder tilt tolerance
        self.tilt_tolerance = 6.0  # degrees

        # Log file
        os.makedirs(os.path.join(script_dir, "../session_log"), exist_ok=True)
        self.log_file = open(os.path.join(script_dir, "../session_log/shoulder_tilt.txt"), "a", encoding="utf-8")

        # toggle fullscreen
        toggle_fullscreen_action = QAction("Toggle Fullscreen", self)
        toggle_fullscreen_action.setShortcut("F11")
        toggle_fullscreen_action.triggered.connect(self.toggle_fullscreen)
        self.addAction(toggle_fullscreen_action)

        # MENU BAR

        # menu: open folder

        # Settings object (stores values in platform‑native location)
        self.settings = QSettings("MyCompany", "ShoulderTiltDashboard")

        # Load last folder if exists
        self.source_folder = self.settings.value("source_folder", "")

        # Menu
        file_menu = self.menuBar().addMenu("File")

        # Sub-Menu: Open Folder
        open_folder_action = QAction("Open Folder", self)
        open_folder_action.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(open_folder_action)

        # Sub-Menu: Open single file
        open_file_action = QAction("Open File", self)
        open_file_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_file_action)

        # menu: view
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(toggle_fullscreen_action)

        # pane
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top layout split
        top = QHBoxLayout()

        # Left side: raw image + dropdown
        left_layout = QVBoxLayout()
        self.proc_label = ClickableLabel()
        self.proc_label.setAlignment(Qt.AlignCenter)
        # self.proc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.proc_label.setFixedSize(512, 512)
        self.proc_label.parent_dash = self

        self.view_selector = QComboBox()
        self.view_selector.addItems(["Unlabeled", "Front", "Diagonal", "Side"])
        self.view_selector.setCurrentIndex(0)
        self.view_selector.setFixedWidth(150)
        self.view_selector.hide()
        self.view_selector.currentIndexChanged.connect(self.on_view_selected)

        left_layout.addWidget(self.proc_label)
        left_layout.addWidget(self.view_selector, alignment=Qt.AlignCenter)


        self.hint_label = QLabel("💡 Esc: reset clicks & reload raw image. F11: open/close fullscreen.")
        self.hint_label.setStyleSheet("color: gray; font-size: 10pt;")
        left_layout.addWidget(self.hint_label)

        # Right side: result image
        self.result_label = QLabel("Result")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        top.addLayout(left_layout, 1)
        top.addWidget(self.result_label, 1)

        # Bottom log panel
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        # self.log.setFixedHeight(200)
        self.log.setFocusPolicy(Qt.NoFocus)

        layout.addLayout(top)
        layout.addWidget(self.log)

        # State
        self.clicks = []
        self.raw_img = None
        self.filename = None
        # self.files = self.load_images_from_folder(self.source_folder)
        # self.load_images_from_folder(self.source_folder)

        self.source_folder = shoulder_tilt_input
        self.load_images_from_folder(self.source_folder)

        self.add_log(f"📂 Using default folder: {self.source_folder}")

        self.index = 0
        self.processed = False
        self.clicks = []
        self.last_tilt = None

        # Resume from progress_shoulder_tilt.txt
        if os.path.exists(os.path.join(script_dir, "../progress/shoulder_tilt.txt") ):
            with open(os.path.join(script_dir, "../progress/shoulder_tilt.txt"), "r", encoding="utf-8") as f:
                last_name = f.read().strip()
            base_names = [os.path.basename(p) for p in self.files]
            if last_name in base_names:
                pos = base_names.index(last_name)
                self.index = min(pos + 1, len(self.files) - 1)
                self.add_log(f"Resuming after {last_name} → starting at image {self.index+1} of {len(self.files)}")

        # Load first image
        if self.files:
            self.load_image(self.files[self.index])
        else:
            self.add_log("No images found in input_images (or all processed).")

        self.statusBar().showMessage("Press Esc to reset clicks and reload the raw image")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder", self.source_folder or os.path.expanduser("~"))
        if folder:
            self.source_folder = folder
            self.settings.setValue("source_folder", folder)
            self.add_log(f"📂 Source folder set to: {folder}")
            self.load_images_from_folder(folder)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            self.source_folder or os.path.expanduser("~"),
            "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            # Reset file list to just this one file
            self.files = [file_path]
            self.index = 0
            self.source_folder = os.path.dirname(file_path)
            self.settings.setValue("source_folder", self.source_folder)
            self.add_log(f"📂 Single file selected: {file_path}")
            self.load_image(file_path)
            self.setWindowTitle("Shoulder Tilt Dashboard — Single File Mode")

    def load_images_from_folder(self, folder=None):
        folder = folder or self.source_folder or "input_images"
        if not folder:
            self.add_log("⚠️ No source folder selected. Use File → Open Folder.")
            self.files = []
            return

        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic"))]
        if not files:
            self.add_log(f"⚠️ No .jpg or .png files found in {folder}")
            self.files = []
            return

        self.files = [os.path.join(folder, f) for f in sorted(files)]
        self.setWindowTitle(f"Shoulder Tilt Dashboard — Bulk Mode ({len(self.files)} images)")
        self.index = 0
        self.add_log(f"✅ Found {len(self.files)} images in {folder}")
        self.load_image(self.files[self.index])
    def show_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.add_log(f"⚠️ Failed to load image: {path}")
            return
        self.proc_label.setPixmap(pixmap.scaled(
            self.proc_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.add_log(f"🖼️ Displaying: {os.path.basename(path)}")

    # logging
    def add_log(self, text):
        # Insert at the top of the QTextEdit
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.Start)
        cursor.insertText(text + "\n")

        # Still append to the session log file in ascending order
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    def closeEvent(self, event):
        if self.log_file:
            self.log_file.close()
        super().closeEvent(event)

    def load_image(self, filename):
        self.filename = filename
        img = cv2.imread(filename)
        # self.img = img
        self.raw_img = img.copy()
        
        if img is None:
            self.add_log(f"⚠️ Could not load {filename}")
            return

        self.add_log(
            f"🖼️ Processing: {os.path.basename(filename)}, {self.index+1}-th of {len(self.files)} files "
        )

        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1)
        
        image = self.raw_img.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize MediaPipe FaceMesh
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:

            # Process the image and get the landmarks
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Key points for the nose
                    # (typically nose tip, left nostril, right nostril)

                    # Index 1 is the tip of the nose
                    nose_tip = face_landmarks.landmark[1]

                    # Index 2 is the left nostril
                    nose_left = face_landmarks.landmark[2]

                    # Index 4 is the right nostril
                    nose_right = face_landmarks.landmark[4]

                    # Index 4 is the nose root
                    nose_root = face_landmarks.landmark[6]

                    # Index 133 is the left eye
                    left_eye = face_landmarks.landmark[133]

                    # Index 362 is the right eye
                    right_eye = face_landmarks.landmark[362]

                    # convert normalized coordinates to pixel coordinates
                    image_height, image_width, _ = image.shape
                    self.nose_tip_x = nose_tip_x = int(nose_tip.x * image_width)
                    self.nose_tip_y = nose_tip_y = int(nose_tip.y * image_height)
                    nose_left_x = int(nose_left.x * image_width)
                    nose_left_y = int(nose_left.y * image_height)
                    nose_right_x = int(nose_right.x * image_width)
                    nose_right_y = int(nose_right.y * image_height)
                    left_eye_x = int(left_eye.x * image_width)
                    left_eye_y = int(left_eye.y * image_height)
                    right_eye_x = int(right_eye.x * image_width)
                    right_eye_y = int(right_eye.y * image_height)

                    # nose root
                    self.nose_root_x = nose_root_x = int(nose_root.x * image_width)
                    self.nose_root_y =nose_root_y = int(nose_root.y * image_height)

                    # ---- FACE ORIENTATION VECTOR (RADIX → NOSE TIP) ----
                    dx = nose_tip_x - nose_root_x
                    dy = nose_tip_y - nose_root_y

                    # Angle of face orientation
                    face_angle = math.atan2(dy, dx)

                    # ---- DRAW NOSE LINE ALONG FACE ORIENTATION ----
                    line_length = 150

                    line_p1 = (
                        nose_tip_x + line_length * math.cos(face_angle),
                        nose_tip_y + line_length * math.sin(face_angle)
                    )

                    line_p2 = (
                        nose_tip_x - line_length * math.cos(face_angle),
                        nose_tip_y - line_length * math.sin(face_angle)
                    )

                    cv2.line(
                        image,
                        (int(line_p1[0]), int(line_p1[1])),
                        (int(line_p2[0]), int(line_p2[1])),
                        (0, 255, 0),
                        2
                    )

        # Label nose line
        x = int((line_p1[0] + line_p2[0]) * 0.55) + 20
        y = int((line_p1[1] + line_p2[1]) * 0.5)
        clue = "Nose line"
        cv2.putText(image, clue, (x, y),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 3)
        cv2.putText(image, clue, (x, y),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 2)

        x1 = int((line_p1[0] + line_p2[0]) * 0.5) + 10
        y1 = int((line_p1[1] + line_p2[1]) * 0.5)

        cv2.line(
            image,
            (x - 5, y - 5),
            (x1, y1),
            (0, 0, 255),
            2
        )

        cv2.line(
            image,
            (x1, y1),
            (x - 25, y + 3),
            (0, 0, 255),
            2
        )

        cv2.line(
            image,
            (x1, y1),
            (x - 27, y - 8),
            (0, 0, 255),
            2
        )


        self.img = image
        preview = image.copy()
        # instruction
        legend_x, legend_y = 0,0
        clue = "Click midpoint of LEFT shoulder contour!"
        cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 3)
        cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 2)
      
        self.add_log(f"1️⃣: {clue}")

        self.proc_label.setPixmap(
            cvimg_to_qpix(preview).scaled(
                self.proc_label.width(), self.proc_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        self.view_selector.setCurrentIndex(0)
        self.view_selector.hide()
        self.clicks = []
        self.processed = False


    def on_click(self, event):
        pos = event.position().toPoint()

        # store plain coordinates
        self.clicks.append((pos.x(), pos.y()))
        self.add_log(f"Point clicked: {pos.x()}, {pos.y()}")

        step = len(self.clicks)

        if self.img is None:
            return

        # copy image and draw grid
        # img_copy = draw_black_grid(self.raw_img.copy(), spacing_px=40)
        img_copy = self.img.copy()

        # scale factors
        h, w = self.raw_img.shape[:2]
        disp_w, disp_h = self.proc_label.width(), self.proc_label.height()
        scale_x, scale_y = w / disp_w, h / disp_h

        # draw all clicked points so far
        coords = []
        for i, (x, y) in enumerate(self.clicks):
            cx, cy = int(x * scale_x), int(y * scale_y)
            coords.append((cx, cy))

            if i == 0:
                cv2.circle(img_copy, (cx, cy), 5, (0, 255, 0), -1)
            if i == 1:
                cv2.circle(img_copy, (cx, cy), 5, (0, 255, 0), -1)
            if i == 2:
                cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), -1)
            if i == 3:
                cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), -1)

            cv2.putText(img_copy, f"{i+1}", (cx, cy+13),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 3)
            cv2.putText(img_copy, f"{i+1}", (cx, cy+13),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 2)

        if step >= 1:
            self.add_log ("✅ Midline of LEFT shoulder recorded.")

            if step == 1:
                # instruction
                legend_x, legend_y = 0,0
                preview = img_copy
                clue = "Click end of LEFT shoulder contour!"
                cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 3)
                cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 2)

                self.add_log(f"2️⃣: {clue}")

        if step >= 2:
            self.add_log("✅ End of RIGHT neck contour recorded.")

            self.p1, self.p2 = p1,p2 = coords[0], coords[1]

            cv2.line(img_copy, p1,p2,(0,255,0), 1)

            # Compute the direction vector from p1 to p2
            direction = np.array([p2[0] - p1[0], p2[1] - p1[1]])

            # Normalize the direction vector (unit vector)
            length = np.linalg.norm(direction)
            direction = direction / length

            # Extend the line by a certain factor (e.g., 100 pixels)
            extend_length = 500

            # Calculate the new extended points
            extended_p1 = (int(p1[0] - direction[0] * extend_length), int(p1[1] - direction[1] * extend_length))
            extended_p2 = (int(p2[0] + direction[0] * extend_length), int(p2[1] + direction[1] * extend_length))

            # Draw the extended line
            cv2.line(img_copy, extended_p1, extended_p2, (0, 255, 0), 2)


            if step == 2:
                # instruction
                legend_x, legend_y = 0,0
                preview = img_copy
                clue = "Click midline of RIGHT shoulder contour!"
                cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 3)
                cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 2)

                self.add_log(f"2️⃣: {clue}")

        if step >= 3:
            self.add_log ("✅ Midline of RIGHT shoulder recorded.")

            if step == 3:
                # instruction
                legend_x, legend_y = 0,0
                preview = img_copy
                clue = "Click end of RIGHT shoulder contour!"
                cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 3)
                cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 2)

                self.add_log(f"3️⃣: {clue}")

        if step >= 4:
            self.add_log("✅ End of RIGHT shoulder contour recorded.")

            self.p3, self.p4 = p3,p4 = coords[2], coords[3]

            cv2.line(img_copy, p3, p4, (0,255,0), 1)

            # Compute the direction vector from p1 to p2
            direction = np.array([p4[0] - p3[0], p4[1] - p3[1]])

            # Normalize the direction vector (unit vector)
            length = np.linalg.norm(direction)
            direction = direction / length

            # Extend the line by a certain factor (e.g., 100 pixels)
            extend_length = 500

            # Calculate the new extended points
            extended_p3 = (int(p3[0] - direction[0] * extend_length), int(p3[1] - direction[1] * extend_length))
            extended_p4 = (int(p4[0] + direction[0] * extend_length), int(p4[1] + direction[1] * extend_length))

            # Draw the extended line
            cv2.line(img_copy, extended_p3, extended_p4, (0, 255, 0), 2)

            # labelling first line
            lbl = "Shoulder line"
            cv2.putText(img_copy, lbl, (int((self.clicks[0][0] + self.clicks[2][0]) * 0.48), self.clicks[0][1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 3)
            cv2.putText(img_copy, lbl, (int((self.clicks[0][0] + self.clicks[2][0]) * 0.48), self.clicks[0][1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 2)
            
            if step == 4:
                # 1. Define the Nose Vector (Reference)
                # Using the coordinates calculated in load_image

                n_dx = self.nose_tip_x - self.nose_root_x
                n_dy = self.nose_tip_y - self.nose_root_y
                nose_angle = math.atan2(n_dy, n_dx)

                # 2. Left Shoulder Angle (Points 1 and 2)
                l_dx = coords[1][0] - coords[0][0]
                l_dy = coords[1][1] - coords[0][1]
                left_sh_angle = math.atan2(l_dy, l_dx)

                # 3. Right Shoulder Angle (Points 3 and 4)
                r_dx = coords[3][0] - coords[2][0]
                r_dy = coords[3][1] - coords[2][1]
                right_sh_angle = math.atan2(r_dy, r_dx)

                # 4. Calculate relative angles to the nose
                # We add/subtract 90 degrees (pi/2) because shoulders are 
                # ideally perpendicular to the nose line
                left_rel_angle = abs(math.degrees(left_sh_angle - nose_angle))
                right_rel_angle = abs(180 - abs(math.degrees(right_sh_angle - nose_angle))) 

                # Normalize to focus on the deviation from "flat" (90 degrees)
                left_final = abs(left_rel_angle - 90)
                right_final = abs(90-right_rel_angle)

                self.left_shoulder_angle = left_final
                self.right_shoulder_angle = right_final

                # Overlay on the image
                # cv2.putText(img_copy, f"Left Sh. Angle: {left_final:.1f} deg", (50, 150), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(img_copy, f"Right Sh. Angle: {right_final:.1f} deg", (50, 180), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # update preview
        self.proc_label.setPixmap(
            cvimg_to_qpix(img_copy).scaled(
                self.proc_label.width(), self.proc_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
         )

        
        # If two clicks, process them
        if len(self.clicks) == 4:
            self.process_points()
            self.view_selector.show()
            # self.clicks = []

    def process_points(self):

        print("run process_points")

        # img = self.raw_img.copy()
        img = self.img
        h, w = img.shape[:2]
        disp_w, disp_h = self.proc_label.width(), self.proc_label.height()
        scale_x, scale_y = w / disp_w, h / disp_h

        def to_coords(p):  # p is a tuple (x, y)
            return int(p[0] * scale_x), int(p[1] * scale_y)

        for i, (x, y) in enumerate(self.clicks):
            cx, cy = int(x * scale_x), int(y * scale_y)
            cv2.putText(img, f"{i+1}", (cx, cy+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 1)

        p1, p2, p3, p4 = [to_coords(p) for p in self.clicks]

        # Draw circles at shoulder line
        cv2.circle(img, p1, 6, (0, 0, 255), -1)
        cv2.circle(img, p2, 6, (0, 0, 255), -1)
        cv2.circle(img, p3, 6, (0, 0, 255), -1)
        cv2.circle(img, p4, 6, (0, 0, 255), -1)

        # Draw shoulder line (red)
        # cv2.line(img, p1, p2, (0, 0, 255), 2)
        # cv2.line(img, p3, p4, (0, 0, 255), 2)


        # Compute the direction vector from p1 to p2
        direction = np.array([p2[0] - p1[0], p2[1] - p1[1]])

        # Normalize the direction vector (unit vector)
        length = np.linalg.norm(direction)
        direction = direction / length

        # Extend the line by a certain factor (e.g., 100 pixels)
        extend_length = 500

        # Calculate the new extended points
        extended_p1 = (int(p1[0] - direction[0] * extend_length), int(p1[1] - direction[1] * extend_length))
        extended_p2 = (int(p2[0] + direction[0] * extend_length), int(p2[1] + direction[1] * extend_length))

        # Draw the extended line
        cv2.line(img, extended_p1, extended_p2, (255, 0, 0), 2)

        # labelling first line
        lbl = "Shoulder line"
        cv2.putText(img, lbl, (p1[0] - 150, p1[1]), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 3)
        cv2.putText(img, lbl, (p1[0] - 150, p1[1]), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 2)

        # labelling second line
        lbl = "Shoulder line"
        cv2.putText(img, lbl, (p3[0], p3[1]), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 3)
        cv2.putText(img, lbl, (p3[0], p3[1]), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 2)

        # Compute the direction vector from p1 to p2
        direction = np.array([p4[0] - p3[0], p4[1] - p3[1]])

        # Normalize the direction vector (unit vector)
        length = np.linalg.norm(direction)
        direction = direction / length

        # Extend the line by a certain factor (e.g., 100 pixels)
        extend_length = 500

        # Calculate the new extended points
        extended_p3 = (int(p3[0] - direction[0] * extend_length), int(p3[1] - direction[1] * extend_length))
        extended_p4 = (int(p4[0] + direction[0] * extend_length), int(p4[1] + direction[1] * extend_length))

        # Draw the extended line
        cv2.line(img, extended_p3, extended_p4, (0, 255, 255), 2)

        # Classification
        if self.left_shoulder_angle >= 16 and self.right_shoulder_angle >=16:
            if abs(self.left_shoulder_angle - self.right_shoulder_angle) <= self.tilt_tolerance:
                folder = output_folders["high"]
                classification = "High ( >= 16 )-Sym"
            else:
                folder = output_folders["high_asymmetric"]
                classification = "High ( >= 16 )-Asym"
        elif 12 < self.left_shoulder_angle < 16 and 12 < self.right_shoulder_angle <16:
            if abs(self.left_shoulder_angle - self.right_shoulder_angle) <= self.tilt_tolerance:
                folder = output_folders["mid"]
                classification = "Mid (12 < ratio < 16 )-Sym"
            else:
                folder = output_folders["mid_asymmetric"]
                classification = "Mid (12 < ratio < 16 )-Asym"
        elif self.left_shoulder_angle <= 12 and self.right_shoulder_angle <=12:
            if abs(self.left_shoulder_angle - self.right_shoulder_angle) <= self.tilt_tolerance:
                folder = output_folders["low"]
                classification = "Low ( < 12 )-Sym"
            else:
                folder = output_folders["low_asymmetric"]
                classification = "Low ( < 12 )-Asym"
        else:
            folder = output_folders["asymmetric_x"]
            classification = "X-asymmetric" # cross-asymmetry = asymmetrical shoulders that cross over categories

        # Legend box
        img_copy = img
        legend_x, legend_y = 0,0
        cv2.rectangle(img_copy, (legend_x-10, legend_y-0),
                      (legend_x + 250, legend_y + 175), (0,0,0), -1)
        cv2.rectangle(img_copy, (legend_x-10, legend_y-0),
                      (legend_x + 250, legend_y + 175), (255,255,255), 1)

        cv2.putText(img_copy, "Legend:", (legend_x, legend_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img_copy, " Green = Nose line", (legend_x, legend_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(img_copy, " Blue = Left shoulder line", (legend_x, legend_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.putText(img_copy, " Yellow = Right shoulder line", (legend_x, legend_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # Classification
        cv2.putText(img_copy, f" Left Degree: {self.left_shoulder_angle:.2f}", (legend_x, legend_y + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.putText(img_copy, f" Right Degree: {self.right_shoulder_angle:.2f}", (legend_x, legend_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        cv2.putText(img_copy, f" Result: {classification}", (legend_x, legend_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(img_copy, f" Time: {timestamp}", (legend_x, legend_y + 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        self.result_label.setPixmap(
            cvimg_to_qpix(img).scaled(
                self.result_label.width(), self.result_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        # Estimate perspective
        perspective = classify_perspective(p1, p2)

        # Save output image
        out_path = os.path.join(folder + "/" + perspective + "_view", os.path.basename(self.filename))

        os.makedirs( os.path.join(folder + "/" + perspective + "_view") , exist_ok=True)
        print("Saving to:", out_path)
        cv2.imwrite(out_path, img_copy)
        self.last_saved_path = out_path
        self.add_log(f"📄 Path: {os.path.basename(self.filename)}")
        self.add_log(f"↙️ Left Tilt: {self.left_shoulder_angle:.2f}°")
        self.add_log(f"↘️ Right Tilt: {self.right_shoulder_angle:.2f}°")
        self.add_log(f"📁 Saved to {out_path}")

        # make progress folder if not exist
        os.makedirs(os.path.join(script_dir, "progress"), exist_ok=True)

        with open(os.path.join(script_dir, "../progress/shoulder_tilt.txt"), "w", encoding="utf-8") as f:
            f.write(os.path.basename(self.filename))

        self.last_tilt = tilt
        self.processed = True


        """
        cv2.putText(img_copy, "1", (p1[0]-5, p1[1]+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img_copy, "1", (p1[0]-5, p1[1]+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(img_copy, "2", (p2[0]-5, p2[1]+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img_copy, "2", (p2[0]-5, p2[1]+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Infer nose anchor (midpoint above shoulders)
        nose_x = (p1[0] + p2[0]) // 2
        nose_y = min(p1[1], p2[1]) - 60
        neck_p1 = (nose_x, nose_y)
        neck_p2 = (nose_x, max(p1[1], p2[1]))

        # Draw upright line (blue)
        cv2.line(img_copy, neck_p1, neck_p2, (255, 0, 0), 2)

        # Calculate tilt using true method
        tilt = calculate_true_tilt(p1, p2)
        ratio = tilt

        # --- Perspective classification ---
        perspective = classify_perspective(p1, p2)

        # Update combo box automatically
        index_map = {"Unlabeled": 0, "Front": 1, "Diagonal": 2, "Side": 3}
        self.view_selector.setCurrentIndex(index_map[perspective])
        self.add_log(f"🧭 Auto-perspective : {perspective}, 👁️ Please compare with your own visual judgment!")

        # Classification
        if ratio >= 16:
            folder = output_folders["high"]
            classification = "High ( >= 16 )"
        elif 12 < ratio < 16:
            folder = output_folders["mid"]
            classification = "Mid (12 < ratio < 16 )"
        elif ratio <= 12:
            folder = output_folders["low"]
            classification = "Narrow ( < 12 )"
        else:
            self.add_log("⚠️ Ratio out of range. Skipped.")
            return

        # Legend box
        legend_x, legend_y = 0,0
        cv2.rectangle(img_copy, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+110), (0,0,0), -1)
        cv2.rectangle(img_copy, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+110), (255,255,255), 1)

        cv2.putText(img_copy, "Legend:", (legend_x, legend_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img_copy, "Red = Shoulder line", (legend_x, legend_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.putText(img_copy, "Blue = Nose line", (legend_x, legend_y+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)



        # Classification
        cv2.putText(img_copy, f"Result: {classification}", (legend_x, legend_y+55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.putText(img_copy, f"Left Degree: {self.left_shoulder_angle}", (legend_x, legend_y+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.putText(img_copy, f"Right Degree: {self.right_shoulder_angle}", (legend_x, legend_y+95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(img_copy, f"Time: {timestamp}", (legend_x, legend_y+95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        """
        # save image result
        """
        if tilt >= 16:
            out_path = os.path.join(folder + "/" + perspective + "_view", os.path.basename(self.filename))
            self.add_log(f"📐 Tilt: {tilt:.2f}°")
        elif tilt >12 and tilt < 16:
            out_path = os.path.join(folder + "/" + perspective + "_view", os.path.basename(self.filename))
            self.add_log(f"📐 Tilt: {tilt:.2f}°")
        elif tilt >=0 and tilt <= 12:
            out_path = os.path.join(folder + "/" + perspective + "_view" , os.path.basename(self.filename))
            self.add_log(f"📐 Tilt: {tilt:.2f}°")
        else:
            self.add_log(f"⚠️ Tilt {tilt:.2f} not in save range. Skipped.")
            self.processed = True
            return
        """

        # out_path = os.path.join(folder + "/" + perspective + "_view", os.path.basename(self.filename))

        # os.makedirs( os.path.join(folder + "/" + perspective + "_view") , exist_ok=True)

        # cv2.imwrite(out_path, img_copy)
        # self.last_saved_path = out_path
        # self.add_log(f"✅ {os.path.basename(self.filename)} → Tilt: {tilt:.2f}° → Saved to {out_path}")

        # make progress folder if not exist
        # os.makedirs(os.path.join(script_dir, "progress"), exist_ok=True)

        # with open(os.path.join(script_dir, "../progress/shoulder_tilt.txt"), "w", encoding="utf-8") as f:
        #     f.write(os.path.basename(self.filename))

        # self.last_tilt = tilt
        # self.processed = True

    def on_view_selected(self, index):
        if index == 0 or not self.processed:
            return

        view_angle = self.view_selector.currentText()
        self.add_log(f"🧭 Perspective: {view_angle}")

        tilt = self.last_tilt
        if tilt is None:
            self.add_log("⚠️ No tilt value available.")
            return

        old_path = self.last_saved_path

        # check for file referred to by old_path
        if not os.path.exists(old_path):
            self.add_log("⚠️ Saved file not found on disk.")
            return

        # Replace old perspective with new one
        new_path = None
        for key in ["Front","Diagonal","Side"]:
            if key in old_path:
                new_path = old_path.replace(key, view_angle)
                break

        if new_path is None:
            # fallback if no keyword found
            folder, fname = os.path.split(old_path)
            new_folder = folder + "_" + "_view"
            os.makedirs(new_folder, exist_ok=True)
            new_path = os.path.join(new_folder, fname)

        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        if old_path != new_path:
            os.replace(old_path, new_path)
            self.add_log(f"✅ File moved to {new_path}")
            self.last_saved_path = new_path
        else:
            self.add_log("ℹ️ File already in correct folder.")

        # Instead of saving Qt pixmap, save annotated OpenCV image
        # cv2.imwrite(out_path, self.raw_img)

        # self.add_log(f"✅ Saved to {out_path}")

        # make progress folder if not exist

        os.makedirs(os.path.join(script_dir, "progress"), exist_ok=True)

        with open(os.path.join(script_dir, "../progress/shoulder_tilt.txt"), "w", encoding="utf-8") as f:
            f.write(os.path.basename(self.filename))

        self.view_selector.setCurrentIndex(0)
        self.view_selector.hide()


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # If a wrong result was already saved, remove it
            if hasattr(self, "last_saved_path") and self.last_saved_path and os.path.exists(self.last_saved_path):
                try:
                    os.remove(self.last_saved_path)
                    self.add_log(f"🗑️ Removed wrong result: {self.last_saved_path}")
                except Exception as e:
                    self.add_log(f"⚠️ Could not remove wrong result: {e}")
                self.last_saved_path = None

            # Reset state and reload current image
            if self.filename:
                self.load_image(self.filename)
                self.add_log("↩️ Reset current image. Please click again.")

            self.clicks = []
            self.processed = False
            self.last_tilt = None
        elif event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Right):
            self.index = min(self.index + 1, len(self.files) - 1)
            self.load_image(self.files[self.index])
        elif event.key() in (Qt.Key_Backspace, Qt.Key_Left):
            self.index = max(self.index - 1, 0)
            self.load_image(self.files[self.index])
        else:
            super().keyPressEvent(event)


    def resizeEvent(self, event):
        # total_w = self.centralWidget().width()
        # total_h = self.centralWidget().height() - self.log.height()
        # half_w = total_w // 2
        # self.proc_label.setFixedSize(half_w, total_h)
        # self.result_label.setFixedSize(half_w, total_h)
        super().resizeEvent(event)

# === Main ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec())

