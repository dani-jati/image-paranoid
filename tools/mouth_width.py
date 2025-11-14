import sys, os, cv2, datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QComboBox
)
from PySide6.QtGui import QPixmap, QImage, QTextCursor, QAction
from PySide6.QtCore import Qt

# === Folders ===
script_dir = os.path.dirname(os.path.abspath(__file__))

cropper_output = os.path.join(script_dir, "../images/output_images/cropper")
mouth_width_input = os.path.join(script_dir, "../images/input_images/mouth_width")

if os.path.isdir(cropper_output) and not os.path.exists(mouth_width_input):
    os.symlink(cropper_output, mouth_width_input)
    print(f"🔗 Symlink created: {mouth_width_input} → {cropper_output}")
elif os.path.islink(mouth_width_input):
    print(f"✅ Symlink already exists: {mouth_width_input} → {os.readlink(mouth_width_input)}")
elif os.path.isdir(mouth_width_input):
    print(f"⚠️ Destination exists as a real folder: {mouth_width_input} — not creating symlink.")
else:
    print(f"⚠️ Cropper output folder missing: {cropper_output}")


print("Symlink points to:", os.readlink(mouth_width_input))

mouth_width_input = os.path.join(script_dir, "../images/input_images/mouth_width")

output_folders = {
    "wide": os.path.join(script_dir, "../images/output_images/mouth_width/1.1_or_more"),
    "mid": os.path.join(script_dir, "../images/output_images/mouth_width/0.9_to_1.1"),
    "narrow": os.path.join(script_dir, "../images/output_images/mouth_width/0.9_or_less"),
}
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

progress_file = os.path.join(script_dir, "../progress/mouth_width.txt")
os.makedirs(os.path.dirname(progress_file), exist_ok=True)

# === Helpers ===
def cvimg_to_qpix(img):
    h, w, ch = img.shape
    bytes_per_line = ch * w
    qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg)

def draw_black_grid(img, spacing_px=40):
    h, w = img.shape[:2]
    for x in range(0, w, spacing_px):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
    for y in range(0, h, spacing_px):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
    return img

def euclidean(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

# === Dashboard ===
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mouth Ratio Dashboard")
        self.resize(1200, 800)

        # Log
        self.log_file = open(os.path.join(script_dir, "../session_log/mouth_width.txt"), "a", encoding="utf-8")
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

        # Left: image
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
        self.files = [os.path.join(mouth_width_input, f) for f in sorted(os.listdir(mouth_width_input)) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
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

        self.proc_label.setPixmap(
            cvimg_to_qpix(self.raw_img).scaled(
                self.proc_label.width(), self.proc_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        self.add_log(f"🖼️ Processing: {os.path.basename(filename)}, {self.index+1}-th of {len(self.files)} files")
        self.clicks = []
        self.add_log("Step 1️⃣: Click center of LEFT pupil.")

    def on_click(self, event):
        pos = event.position().toPoint()

        # store plain coordinates
        self.clicks.append((pos.x(), pos.y()))
        self.add_log(f"Point clicked: {pos.x()}, {pos.y()}")

        step = len(self.clicks)

        if step == 1:
            self.add_log("✅ Left pupil center recorded. Step 2️⃣: Click center of RIGHT pupil.")
        elif step == 2:
            self.add_log("✅ Right pupil recorded. Step 3️⃣: Click NASOLABIAL angle.")

            # --- Draw eye line immediately ---
            img_copy = self.raw_img.copy()
            h, w = self.raw_img.shape[:2]
            disp_w, disp_h = self.proc_label.width(), self.proc_label.height()
            scale_x, scale_y = w / disp_w, h / disp_h

            p1 = (int(self.clicks[0][0] * scale_x), int(self.clicks[0][1] * scale_y))
            p2 = (int(self.clicks[1][0] * scale_x), int(self.clicks[1][1] * scale_y))

            cv2.line(img_copy, p1, p2, (0, 255, 0), 2)
            cv2.circle(img_copy, p1, 4, (0, 255, 0), -1)
            cv2.circle(img_copy, p2, 4, (0, 255, 0), -1)
            cv2.putText(img_copy, "Eye line", (p1[0], p1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            self.proc_label.setPixmap(
                cvimg_to_qpix(img_copy).scaled(
                    self.proc_label.width(), self.proc_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )            

        # Draw preview with numbered points
        img_copy = draw_black_grid(self.raw_img.copy(), spacing_px=40)

        # scale factors
        h, w = self.raw_img.shape[:2]
        disp_w, disp_h = self.proc_label.width(), self.proc_label.height()
        scale_x, scale_y = w / disp_w, h / disp_h

        # Draw all clicked points
        coords = []
        for i, (x, y) in enumerate(self.clicks):
            cx, cy = int(x * scale_x), int(y * scale_y)
            print(f"Click {i}: ({cx}, {cy})")
            coords.append((cx, cy))
            if i == 0:
                cv2.circle(img_copy, (cx, cy), 5, (0, 255, 0), -1)
            if i == 1:
                cv2.circle(img_copy, (cx, cy), 5, (0, 255, 0), -1)
            if i == 2:
                cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), -1)
            if i == 3:
                cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)
            if i == 4:
                cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img_copy, f"{i+1}", (cx+5, cy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.circle(img_copy, (cx, cy), 5, color, -1)  # small dot

        if step >= 2:
            self.add_log("draw line from p1 to p2")
            p1,p2 = coords[0], coords[1]
            cv2.line(img_copy, p1, p2, (0,255,0), 1)
            cv2.putText(img_copy, "Eye line", (p1[0], p1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            p3 = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

            # Midpoint
            cv2.circle(img_copy, p3, 4, (255, 0, 0), -1)

        if step >=3:
            p4 = coords[2]
            cv2.line(img_copy, p3, p4, (255, 0, 0), 2)
            cv2.putText(img_copy, "Nose line", (p4[0], p4[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if step == 3:
            self.add_log("✅ Nasolabial angle recorded. Step 4️⃣: Click LEFT mouth corner.")

        if step == 4:
            self.add_log("✅ Left mouth corner recorded. Step 5️⃣: Click RIGHT mouth corner.")

        if step >=5:
            p5 = coords[3]
            p6 = coords[4]

            # mouth line
            cv2.line(img_copy, p5, p6, (0, 0, 255), 2)

            # draw small circle at mount-line start
            cv2.circle(img_copy, p5, 5, (0, 0, 255), -1)

            # draw small circle at mount-line end
            cv2.circle(img_copy, p6, 5, (0, 0, 255), -1)

            # labelling mouth line
            cv2.putText(img_copy, "Mouth line", (p5[0], p5[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            self.add_log("✅ Right mouth corner recorded. Step 6️⃣: Confirming all points…")

        for i, (x, y) in enumerate(self.clicks):
            cx, cy = int(x * scale_x), int(y * scale_y)
            # cv2.circle(img_copy, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(img_copy, f"{i+1}", (cx, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 1)


        self.proc_label.setPixmap(
            cvimg_to_qpix(img_copy).scaled(
                self.proc_label.width(), self.proc_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        self.process_clicks()


    def process_clicks(self):
        img = self.raw_img.copy()
        h, w = img.shape[:2]
        disp_w, disp_h = self.proc_label.width(), self.proc_label.height()
        scale_x, scale_y = w / disp_w, h / disp_h

        def to_coords(p):
            return int(p[0] * scale_x), int(p[1] * scale_y)
        if len(self.clicks) >= 5:
            p1, p2, p4, p5, p6 = [to_coords(p) for p in self.clicks[:4]] + [to_coords(self.clicks[4])]
        else:
            # print("Not enough clicks! Need at least 5.")
            return
        p3 = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

        # Eye line
        cv2.line(img, p1, p2, (0, 255, 0), 2)

        # Draw small circles at first-line start
        cv2.circle(img, p1, 5, (0, 255, 0), -1)

        # Draw small circles at first-line end
        cv2.circle(img, p2, 5, (0, 255, 0), -1)

        # labelling eye line
        cv2.putText(img, "Eye line", (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Midpoint
        cv2.circle(img, p3, 4, (255, 0, 0), -1)

        # labelling midpoint
        cv2.putText(img, "Midpoint", (p3[0]+5, p3[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        # Nose line
        cv2.line(img, p3, p4, (255, 0, 0), 2)

        # Draw small circles at nose-line end
        cv2.circle(img, p4, 5, (255, 0, 0), -1)

        cv2.putText(img, "Nose line", (p4[0], p4[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,200), 1)
        d_nose = euclidean(p3, p4)

        # Mouth line
        cv2.line(img, p5, p6, (0, 0, 255), 2)

        # Draw small circles at mouth-line start
        cv2.circle(img, p5, 5, (0, 0, 255), -1)

        # Draw small circles at mouth-line end
        cv2.circle(img, p6, 5, (0, 0, 255), -1)

        # labelling mouth line
        cv2.putText(img, "Mouth line", (p5[0], p5[1]+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        d_mouth = euclidean(p5, p6)


        for i, (x, y) in enumerate(self.clicks):
            cx, cy = int(x * scale_x), int(y * scale_y)
            # cv2.circle(img_copy, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(img, f"{i+1}", (cx, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 1)

        # Ratio
        ratio = d_mouth / d_nose if d_mouth else 0
        self.add_log(f"📏 Nose length: {d_nose:.2f}, Mouth width: {d_mouth:.2f}, Ratio: {ratio:.2f}")

        # Classification
        if ratio > 1.1:
            folder = output_folders["wide"]
            classification = "Wide ( > 1.1 )"
        elif 0.9 <= ratio <= 1.1:
            folder = output_folders["mid"]
            classification = "Mid ( 0.9 - 1.1 )"
        elif 0.0 < ratio < 0.9:
            folder = output_folders["narrow"]
            classification = "Narrow ( < 0.9 )"
        else:
            self.add_log("⚠️ Ratio out of range. Skipped.")
            return

        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Legend box
        legend_x, legend_y = 0, 0
        cv2.rectangle(img, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+150), (0,0,0), -1)
        cv2.rectangle(img, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+150), (255,255,255), 1)

        cv2.putText(img, "Legend:", (legend_x, legend_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img, "Green = Eye line", (legend_x, legend_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(img, "Yellow = Midpoint", (legend_x, legend_y+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(img, "Blue = Nose line", (legend_x, legend_y+55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.putText(img, "Red = Mouth line", (legend_x, legend_y+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # Classification + ratio + timestamp
        cv2.putText(img, f"Result: {classification}", (legend_x, legend_y+95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img, f"Ratio: {ratio:.2f}", (legend_x, legend_y+115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img, f"Time: {timestamp}", (legend_x, legend_y+135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # Image result
        out_path = os.path.join(folder, os.path.basename(self.filename))
        
        cv2.imwrite(out_path, img)
        self.last_saved_path = out_path

        self.result_label.setPixmap(cvimg_to_qpix(img).scaled(
            self.result_label.width(), self.result_label.height(),
            Qt.KeepAspectRatio))

        self.add_log(f"✅ Saved to {out_path}")

        # Logging progress
        open(progress_file, "w").write(os.path.basename(self.filename))

    def add_log(self, text):
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.Start)
        cursor.insertText(text + "\n")
        self.log_file.write(text + "\n")
        self.log_file.flush()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if hasattr(self, "last_saved_path") and self.last_saved_path and os.path.exists(self.last_saved_path):
                try:
                    os.remove(self.last_saved_path)
                    self.add_log(f"🗑️ Removed wrong result: {self.last_saved_path}")
                except Exception as e:
                    self.add_log(f"⚠️ Could not remove wrong result: {e}")
                self.last_saved_path = None
            if self.filename:
                self.load_image(self.filename)
                self.add_log("↩️ Reset current image. Please click again.")
            self.clicks = []
        elif event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Right):
            self.index = min(self.index + 1, len(self.files) - 1)
            self.load_image(self.files[self.index])
        elif event.key() in (Qt.Key_Backspace, Qt.Key_Left):
            self.index = max(self.index - 1, 0)
            self.load_image(self.files[self.index])
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.log_file.close()
        super().closeEvent(event)


# === Main ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec())

