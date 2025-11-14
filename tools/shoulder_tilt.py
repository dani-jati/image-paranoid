import sys, os, cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QSizePolicy, QComboBox, QFileDialog
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QImage, QTextCursor, QPainter, QColor, QFont, QAction
from PySide6.QtCore import QSettings

# === Folders ===
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "../images/input_images/shoulder_tilt")

output_folder_high = os.path.join(script_dir, "../images/output_images/shoulder_tilt/16_or_above")
output_folder_mid  = os.path.join(script_dir, "../images/output_images/shoulder_tilt/12.1_to_15.9")
output_folder_low  = os.path.join(script_dir, "../images/output_images/shoulder_tilt/0_to_12")

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder_high, exist_ok=True)
os.makedirs(output_folder_mid, exist_ok=True)
os.makedirs(output_folder_low, exist_ok=True)

# Ensure symlink from cropper output to shoulder-tilt input
cropper_output = os.path.join(script_dir, "../images/output_images/cropper")
shoulder_tilt_input = os.path.join(script_dir, "../images/input_images/shoulder_tilt")

if os.path.isdir(cropper_output) and not os.path.exists(shoulder_tilt_input):
    os.symlink(cropper_output, shoulder_tilt_input)
    print(f"🔗 Symlink created: {shoulder_tilt_input} → {cropper_output}")
elif os.path.islink(shoulder_tilt_input):
    print(f"✅ Symlink already exists: {shoulder_tilt_input} → {os.readlink(shoulder_tilt_input)}")
elif os.path.isdir(shoulder_tilt_input):
    print(f"⚠️ Destination exists as a real folder: {shoulder_tilt_input} — not creating symlink.")
else:
    print(f"⚠️ Cropper output folder missing: {cropper_output}")

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
    processed = set(os.listdir(output_folder_high)) | set(os.listdir(output_folder_low))
    files = [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
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
            self.parent_dash.register_click(event.pos())

# === Dashboard ===
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shoulder Tilt Dashboard")
        self.resize(1200, 800)

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
        self.current_img = None
        self.filename = None
        # self.files = self.load_images_from_folder(self.source_folder)
        # self.load_images_from_folder(self.source_folder)

        self.source_folder = input_folder
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
        self.current_img = cv2.imread(filename)

        if self.current_img is None:
            self.add_log(f"⚠️ Could not load {filename}")
            return

        self.proc_label.setPixmap(
            cvimg_to_qpix(self.current_img).scaled(
                self.proc_label.width(), self.proc_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        self.add_log(
            f"🖼️ Processing: {os.path.basename(filename)}, {self.index+1}-th of {len(self.files)} files "
        )

        self.clicks = []
        self.processed = False
        self.view_selector.setCurrentIndex(0)
        self.view_selector.hide()

    def register_click(self, pos):
        self.clicks.append(pos)
        self.add_log(f"Point clicked: {pos.x()}, {pos.y()}")

        if self.current_img is None:
            return

        # Get display pixmap and widget size
        pixmap = self.proc_label.pixmap()
        if pixmap is None:
            return

        disp_w, disp_h = pixmap.width(), pixmap.height()
        lbl_w, lbl_h = self.proc_label.width(), self.proc_label.height()

        # Compute offsets (letterboxing margins)
        offset_x = (lbl_w - disp_w) / 2
        offset_y = (lbl_h - disp_h) / 2

        # Translate click into pixmap coordinates
        px = pos.x() - offset_x
        py = pos.y() - offset_y

        if px < 0 or py < 0 or px > disp_w or py > disp_h:
            self.add_log("⚠️ Click outside image area")
            return

        # Scale to original image coordinates
        h, w = self.current_img.shape[:2]
        scale_x, scale_y = w / disp_w, h / disp_h
        cx, cy = int(px * scale_x), int(py * scale_y)

        # Draw dots for all clicks so far
        img_copy = self.current_img.copy()
        for i, p in enumerate(self.clicks):
            # Recalculate each click with offset correction
            px = p.x() - offset_x
            py = p.y() - offset_y
            cx, cy = int(px * scale_x), int(py * scale_y)
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.circle(img_copy, (cx, cy), 6, color, -1)

        # Update preview
        self.proc_label.setPixmap(
            cvimg_to_qpix(img_copy).scaled(
                lbl_w, lbl_h,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        # If two clicks, process them
        if len(self.clicks) == 2:
            self.process_points(self.clicks)
            self.view_selector.show()
            self.clicks = []


    def process_points(self, points):
        if len(points) != 2 or self.current_img is None:
            return
        img_copy = self.current_img.copy()
        h, w = img_copy.shape[:2]

        pixmap = self.proc_label.pixmap()
        if pixmap is None:
            return
        disp_w, disp_h = pixmap.width(), pixmap.height()
        label_w, label_h = self.proc_label.width(), self.proc_label.height()
        x_offset = (label_w - disp_w) // 2
        y_offset = (label_h - disp_h) // 2

        adj_x1 = points[0].x() - x_offset
        adj_y1 = points[0].y() - y_offset
        adj_x2 = points[1].x() - x_offset
        adj_y2 = points[1].y() - y_offset

        scale_x, scale_y = w / disp_w, h / disp_h
        p1 = (int(adj_x1 * scale_x), int(adj_y1 * scale_y))
        p2 = (int(adj_x2 * scale_x), int(adj_y2 * scale_y))

        # Draw shoulder line (red)
        cv2.line(img_copy, p1, p2, (0, 0, 255), 2)

        # Infer nose anchor (midpoint above shoulders)
        nose_x = (p1[0] + p2[0]) // 2
        nose_y = min(p1[1], p2[1]) - 60
        neck_p1 = (nose_x, nose_y)
        neck_p2 = (nose_x, max(p1[1], p2[1]))

        # Draw upright line (blue)
        cv2.line(img_copy, neck_p1, neck_p2, (255, 0, 0), 2)

        # Calculate tilt using true method
        tilt = calculate_true_tilt(p1, p2)

        # --- Perspective classification ---
        perspective = classify_perspective(p1, p2)

        # Update combo box automatically
        index_map = {"Unlabeled": 0, "Front": 1, "Diagonal": 2, "Side": 3}
        self.view_selector.setCurrentIndex(index_map[perspective])
        self.add_log(f"🧭 Auto-perspective : {perspective}, 👁️ Please compare with your own visual judgment!")

        # Annotate image
        cv2.putText(img_copy, f"Tilt: {tilt:.2f} deg", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 50), 2)

        self.result_label.setPixmap(
            cvimg_to_qpix(img_copy).scaled(
                self.result_label.width(), self.result_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        if tilt >= 16:
            out_path = os.path.join(output_folder_high + "/" + perspective + "_view", os.path.basename(self.filename))
            self.add_log(f"📐 Tilt: {tilt:.2f}°")
        elif tilt >12 and tilt < 16:
            out_path = os.path.join(output_folder_mid + "/" + perspective + "_view", os.path.basename(self.filename))
            self.add_log(f"📐 Tilt: {tilt:.2f}°")
        elif tilt >=0 and tilt <= 12:
            out_path = os.path.join(output_folder_low + "/" + perspective + "_view" , os.path.basename(self.filename))
            self.add_log(f"📐 Tilt: {tilt:.2f}°")
        else:
            self.add_log(f"⚠️ Tilt {tilt:.2f} not in save range. Skipped.")
            self.processed = True
            return

        os.makedirs( os.path.join(output_folder_high + "/" + perspective + "_view") , exist_ok=True)

        cv2.imwrite(out_path, img_copy)
        self.last_saved_path = out_path
        self.add_log(f"✅ {os.path.basename(self.filename)} → Tilt: {tilt:.2f}° → Saved to {out_path}")

        # make progress folder if not exist
        os.makedirs(os.path.join(script_dir, "progress"), exist_ok=True)

        with open(os.path.join(script_dir, "../progress/shoulder_tilt.txt"), "w", encoding="utf-8") as f:
            f.write(os.path.basename(self.filename))

        self.last_tilt = tilt
        self.processed = True

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
        # cv2.imwrite(out_path, self.current_img)

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

