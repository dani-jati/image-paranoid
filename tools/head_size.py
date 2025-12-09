# Script Purpose:
# Measure head size relative to bishoulder width.

# Definition of Bishoulder Width:
# Bishoulder width is measured between the outermost lateral points of the left and right shoulders.
# In this system, the reference landmark is the acromiohumeral notch (shoulder joint depression / "ujung pundak").

# Instructions to Identify Shoulder Ends:
# • Extend both arms straight out to the sides (left arm to the left, right arm to the right).
# • At the top of each shoulder, palpate for a small depression where the upper arm bone (humerus) meets the shoulder bone (scapula/clavicle).
# • This depression (acromiohumeral notch) marks the most lateral point of each shoulder.
# • Use these points as the reference ends for measuring bishoulder width.

# Clothing Adjustment Rule:
# If the subject is wearing clothing and the notch is not visible,
# estimate the shoulder end position by tracing the contour line of the shoulder.
# The lateral end is located where the contour begins to turn downward toward the arm.

# IF palpation_allowed:
#    LEFT_END = left_acromiohumeral_notch
#    RIGHT_END = right_acromiohumeral_notch
# ELSE:
#     LEFT_END = left_contour_turn_down
#     RIGHT_END = right_contour_turn_down

# BISHOULDER_WIDTH = distance(LEFT_END, RIGHT_END)


import sys, os, cv2, datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QComboBox
)
from PySide6.QtGui import QPixmap, QImage, QTextCursor, QAction
from PySide6.QtCore import Qt

# === Folders ===
script_dir = os.path.dirname(os.path.abspath(__file__))

output_folders = {
    "wide": os.path.join(script_dir, "../images/output_images/head_size/1.65_or_more"),
    "mid": os.path.join(script_dir, "../images/output_images/head_size/1.35_to_1.65"),
    "narrow": os.path.join(script_dir, "../images/output_images/head_size/0_to_1.35"),
}

for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Ensure symlink from cropper output to shoulder-tilt input
cropper_output = os.path.normpath(os.path.join(script_dir, "..", "images", "output_images", "cropper"))
head_size_input = os.path.normpath(os.path.join(script_dir, "..", "images", "input_images", "head_size"))
path = head_size_input

if os.path.isdir(cropper_output) and not os.path.exists(head_size_input):

    if os.name == 'posix': #linux or mac
        os.symlink(cropper_output, head_size_input) #linux
    elif os.name == 'nt': #windows
        os.symlink(cropper_output, head_size_input, target_is_directory=True) #windows

    print(f"🔗 Symlink created: {head_size_input} → {cropper_output}")
elif os.path.islink(head_size_input):
    print(f"✅ Symlink already exists: {head_size_input} → {os.readlink(head_size_input)}")
elif os.path.isdir(head_size_input):
    print(f"⚠️ Destination exists as a real folder: {head_size_input} — not creating symlink.")
else:
    if not os.path.islink(path):
        print(f"{path} is not a symbolic link.")
        os.remove(path)

        if os.name == 'posix': #linux or mac
            os.symlink(cropper_output, head_size_input) #linux
        elif os.name == 'nt': #windows
            os.symlink(cropper_output, head_size_input, target_is_directory=True) #windows

        print(f"🔗 Symlink created: {head_size_input} → {cropper_output}")

print("Symlink points to:", os.readlink(head_size_input))

progress_file = os.path.join(script_dir, "../progress/head_size.txt")
os.makedirs(os.path.dirname(progress_file), exist_ok=True)

# === Helpers ===
def cvimg_to_qpix(img):
    h, w, ch = img.shape
    bytes_per_line = ch * w
    qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg)

def draw_grid(img, spacing=40):
    h, w = img.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
    for y in range(0, h, spacing):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
    return img

def euclidean(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def draw_black_grid(img, spacing_px=40):
    """Draws a black grid spaced by spacing_px pixels."""
    h, w = img.shape[:2]
    for x in range(0, w, spacing_px):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
    for y in range(0, h, spacing_px):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
    return img


def classify_perspective(p1, p2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    if dx < 30:  # head too close, probably aside
        return "Side"

    ratio = dy / dx
    if ratio > 0.4:
        return "Diagonal"
    else:
        return "Front"


# === Dashboard ===
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Head Size Dashboard")
        self.resize(1200, 800)

        # Log
        self.log_file = open(os.path.join(script_dir, "../session_log/head_size.txt"), "a", encoding="utf-8")
        os.makedirs(os.path.dirname(self.log_file.name), exist_ok=True)

        # Fullscreen toggle
        toggle_fullscreen = QAction("Toggle Fullscreen", self)
        toggle_fullscreen.setShortcut("F11")
        toggle_fullscreen.triggered.connect(self.toggle_fullscreen)
        self.addAction(toggle_fullscreen)

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        top = QHBoxLayout()

        # Left: image + dropdown
        left = QVBoxLayout()
        self.proc_label = QLabel()
        self.proc_label.setAlignment(Qt.AlignCenter)
        self.proc_label.setFixedSize(512, 512)
        self.proc_label.mousePressEvent = self.on_click

        self.view_selector = QComboBox()
        self.view_selector.addItems(["Unlabeled", "Front", "Diagonal", "Side"])
        self.view_selector.setFixedWidth(150)
        self.view_selector.hide()

        left.addWidget(self.proc_label)
        left.addWidget(self.view_selector, alignment=Qt.AlignCenter)

        # Right: result
        self.result_label = QLabel("Result")
        self.result_label.setAlignment(Qt.AlignCenter)

        top.addLayout(left, 1)
        top.addWidget(self.result_label, 1)

        # Bottom: log
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addLayout(top)
        layout.addWidget(self.log)

        # State
        self.clicks = []
        self.files = [os.path.join(head_size_input, f) for f in sorted(os.listdir(head_size_input)) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.index = 0
        self.last_saved_path = None

        # Resume
        if os.path.exists(progress_file):
            last = open(progress_file).read().strip()
            base_names = [os.path.basename(p) for p in self.files]
            if last in base_names:
                self.index = min(base_names.index(last) + 1, len(self.files) - 1)

        if self.files:
            self.load_image(self.files[self.index])
        else:
            self.add_log("⚠️ No images found in input folder.")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def load_image(self, filename):
        self.filename = filename
        img = cv2.imread(filename)
        self.raw_img = img.copy()

        if self.raw_img is None:
            self.add_log(f"⚠️ Could not load {filename}")
            return

        """
        self.proc_label.setPixmap(
            cvimg_to_qpix(self.raw_img).scaled(
                self.proc_label.width(), self.proc_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        """

        self.add_log(
            f"🖼️ Processing: {os.path.basename(filename)}, {self.index+1}-th of {len(self.files)} files "
        )

        # instruction
        legend_x, legend_y = 0,0
        preview = img.copy()
        cv2.putText(preview, "Click leftmost point of midface contour!", (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(preview, "Click leftmost point of midface contour!", (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
      
        self.add_log("Click lateral angle of left eye!")

        self.proc_label.setPixmap(cvimg_to_qpix(preview).scaled(self.proc_label.width(), self.proc_label.height(), Qt.KeepAspectRatio))

        self.view_selector.setCurrentIndex(0)
        self.view_selector.hide()
        self.clicks = []


    def on_click(self, event):
        pos = event.position().toPoint()

        # store plain coordinates
        self.clicks.append((pos.x(), pos.y()))
        self.add_log(f"Point clicked: {pos.x()}, {pos.y()}")

        step = len(self.clicks)

        if self.raw_img is None:
            return




        # copy image and draw grid
        # img_copy = draw_black_grid(self.raw_img.copy(), spacing_px=40)
        img_copy = self.raw_img.copy()

        # scale factors
        h, w = self.raw_img.shape[:2]
        disp_w, disp_h = self.proc_label.width(), self.proc_label.height()
        scale_x, scale_y = w / disp_w, h / disp_h

        # draw all points so far
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

            cv2.putText(img_copy, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img_copy, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)

        if step == 1:
            self.add_log("✅ Leftmost point of midface contour recorded.")

            # guide line
            cv2.line(img_copy,(cx-400, cy), (cx+400, cy), (150,150,150), 1)

            # instruction
            legend_x, legend_y = 0,0
            preview = img_copy
            cv2.putText(preview, "Click rightmost point of midface contour!", (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(preview, "Click rightmost point of midface contour!", (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

            self.add_log("2️⃣: Click rightmost point of midface contour!")

        if step >= 2:
            self.add_log("draw line from leftmost to rightmost mid-face contour")

            p1,p2 = coords[0], coords[1]

            cv2.line(img_copy, p1, p2, (0,255,0), 1)

            cv2.putText(img_copy, "Head width", (p1[0], p1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img_copy, "Head width", (p1[0], p1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            if step == 2:
                # instruction
                legend_x, legend_y = 0,0
                preview = img_copy
                cv2.putText(preview, "Click left acromiohumeral notch (left shoulder end)!", (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(preview, "Click left acromiohumeral notch (left shoulder end)!", (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)               

            self.add_log("Click left acromiohumeral notch (left shoulder end)!")

        if step >= 3:

            self.add_log("Left acromiohumeral notch (left shoulder end) recorded.")
            
            if step == 3:
                cv2.line(img_copy,(cx, cy-400), (cx, cy+400), (150,150,150), 1)

                # instruction
                legend_x, legend_y = 0,0
                preview = img_copy
                cv2.putText(preview, "Click right acromiohumeral notch (right shoulder end)!", (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(preview, "Click right acromiohumeral notch (right shoulder end)!", (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)               

                self.add_log("Click right acromiohumeral notch (right shoulder end)!")

        if step >= 4:
            self.add_log("draw bishoulder line")
            p3,p4 = coords[2], coords[3]
            cv2.line(img_copy, p3, p4, (255,0,0), 1)
            cv2.putText(img_copy, "Bishoulder line", (p3[0], p4[1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img_copy, "Bishoulder line", (p3[0], p4[1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Divide second line into 3 parts
            dx = ( p4[0] - p3[0] ) / 3.0
            dy = ( p4[1] - p3[1] ) / 3.0

            div1 = (int(p3[0] + dx), int(p3[1] + dy))
            div2 = (int(p3[0] + 2*dx), int(p3[1] + 2*dy))

            # Draw vertical separator lines
            line_length = 400
            cv2.line(img_copy, (div1[0], div1[1] - line_length), (div1[0], div1[1]), (0, 255, 255), 2)
            cv2.line(img_copy, (div2[0], div2[1] - line_length), (div2[0], div2[1]), (0, 255, 255), 2)

            # Draw small circles at division points
            cv2.circle(img_copy, div1, 5, (0, 255, 255), -1)
            cv2.circle(img_copy, div2, 5, (0, 255, 255), -1)

            # Add labels below the division points
            cv2.putText(img_copy, "1/3", (div1[0] - 10, div1[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(img_copy, "1/3", (div1[0] - 10, div1[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)

            cv2.putText(img_copy, "2/3", (div2[0] - 10, div2[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(img_copy, "2/3", (div2[0] - 10, div2[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)

        # update preview
        self.proc_label.setPixmap(
            cvimg_to_qpix(img_copy).scaled(
                self.proc_label.width(), self.proc_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
         )


        if len(self.clicks) == 4:
            self.process_clicks()

    def process_clicks(self):
        img = self.raw_img.copy()
        h, w = img.shape[:2]
        disp_w, disp_h = self.proc_label.width(), self.proc_label.height()
        scale_x, scale_y = w / disp_w, h / disp_h

        def to_coords(p):  # p is a tuple (x, y)
            return int(p[0] * scale_x), int(p[1] * scale_y)

        p1, p2, p3, p4 = [to_coords(p) for p in self.clicks]

        for i, (x, y) in enumerate(self.clicks):
            cx, cy = int(x * scale_x), int(y * scale_y)

            cv2.putText(img, f"{i+1}", (cx-5, cy+20),   
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)


        # First line
        cv2.line(img, p1, p2, (0, 255, 0), 2)
        cv2.circle(img, p1, 5, (0, 255, 0), -1)
        cv2.circle(img, p2, 5, (0, 255, 0), -1)

        # labelling first line
        cv2.putText(img, "Head line", (p1[0], p1[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(img, "Head line", (p1[0], p1[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


        # Second line
        cv2.line(img, p3, p4, (255, 0, 0), 2)
        cv2.circle(img, p3, 5, (255, 0, 0), -1)
        cv2.circle(img, p4, 5, (255, 0, 0), -1)

        # labelling second line
        cv2.putText(img, "Shoulder line", (p3[0]+20, p4[1]+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(img, "Shoulder line", (p3[0]+20, p4[1]+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Divide second line into 3 parts
        dx = ( p4[0] - p3[0] ) / 3.0
        dy = ( p4[1] - p3[1] ) / 3.0

        line_length = 400

        for i in range(1, 3):  # division points at 1/3 ... 2/3
            div = (int(p3[0] + i * dx), int(p3[1] + i * dy))

            # Draw vertical separator lines
            cv2.line(img, (div[0], div[1] - line_length), (div[0], div[1]), (0, 255, 255), 1)

            # Draw small circles at division points
            cv2.circle(img, div, 5, (0, 255, 255), -1)

            # Add labels below the division points
            cv2.putText(img, f"{i}/3", (div[0] - 10, div[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(img, f"{i}/3", (div[0] - 10, div[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)

        L1 = euclidean(p1, p2)
        L2 = euclidean(p3, p4)
        ratio = ( L1 / L2 ) * 3 if L1 else 0

        # --- Perspective classification ---
        perspective = classify_perspective(p1, p2)

        # Update combo box automatically
        index_map = {"Unlabeled": 0, "Front": 1, "Diagonal": 2, "Side": 3}
        self.view_selector.setCurrentIndex(index_map[perspective])
        self.add_log(f"🧭 Auto-perspective : {perspective}, 👁️ Please compare with your own visual judgment!")

        """
        cv2.putText(img, f"Ratio: {ratio:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        self.add_log(f"📏 L1: {L1:.2f}, L2: {L2:.2f}, Ratio: {ratio:.2f}")
        """



        if ratio > 1.65:
            folder = output_folders["wide"]
            classification = "Wide ( > 1.65 )"
        elif 1.35 <= ratio <= 1.65:
            folder = output_folders["mid"]
            classification = "Mid ( 1.35 - 1.65 )"
        elif 0 <= ratio < 1.35:
            folder = output_folders["narrow"]
            classification = "Narrow ( 0 - 1.35 )"
        else:
            self.add_log("⚠️ Ratio out of range. Skipped.")
            classification = "Out of range"
            return

        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Legend box
        legend_x, legend_y = 0, 0
        cv2.rectangle(img, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+130), (0,0,0), -1)
        cv2.rectangle(img, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+130), (255,255,255), 1)

        cv2.putText(img, "Legend:", (legend_x, legend_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, "Green = Head line", (legend_x, legend_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(img, "Yellow = Thirds", (legend_x, legend_y+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        cv2.putText(img, "Blue = Shoulder line", (legend_x, legend_y+55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        """
        cv2.putText(img, "Red = Mouth line", (legend_x, legend_y+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        """

        # Classification + ratio + timestamp
        cv2.putText(img, f"Result: {classification}", (legend_x, legend_y+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, f"Ratio: {ratio:.2f}", (legend_x, legend_y+95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, f"Time: {timestamp}", (legend_x, legend_y+115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)

        # save result
        out_path = os.path.join(folder, os.path.basename(self.filename))
        cv2.imwrite(out_path, img)
        self.last_saved_path = out_path
        self.result_label.setPixmap(cvimg_to_qpix(img).scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))
        self.add_log(f"✅ Saved to {out_path}")
        open(progress_file, "w").write(os.path.basename(self.filename))
        self.view_selector.show()

    def add_log(self, text):
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.Start)
        cursor.insertText(text + "\n")
        self.log_file.write(text + "\n")
        self.log_file.flush()

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

        """
        if event.key() == Qt.Key_Escape:
            if self.last_saved_path and os.path.exists(self.last_saved_path):
                os.remove(self.last_saved_path)
                self.add_log(f"🗑️ Removed: {self.last_saved_path}")
            self.load_image(self.filename)
        elif event.key() in (Qt.Key_Space, Qt.Key_Right):
            self.index = min(self.index + 1, len(self.files) - 1)
            self.load_image(self.files[self.index])
        elif event.key() in (Qt.Key_Backspace, Qt.Key_Left):
            self.index = max(self.index - 1, 0)
            self.load_image(self.files[self.index])
        else:
            super().keyPressEvent(event)
        """

    def closeEvent(self, event):
        self.log_file.close()
        super().closeEvent(event)


# === Main ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec())


# Bishoulder Width Landmark Definitions
# 1. Palpation Method (Anatomical Landmark)
# • 	Landmark: Acromiohumeral notch (shoulder joint depression / ujung pundak).
# • 	Procedure: Extend both arms sideways. Palpate the small depression where the humerus meets the scapula/clavicle.
# • 	Use: Mark these points as the lateral ends of the shoulders.
# 2. Visual Estimation Method (Contour Landmark)
# • 	Landmark: Contour turn‑down point (outer shoulder curve).
# • 	Procedure: Observe the shoulder outline. Identify the point where the horizontal contour of the shoulder begins to slope downward toward the arm.
# • 	Use: Mark these points as the lateral ends of the shoulders when palpation is not possible (e.g., cultural respect, clothing).