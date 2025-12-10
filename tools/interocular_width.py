# This script is to measure interocular width 
# relative to width between outer side of both lateral eye angles
# interocular width is distance in-between the most medial eye angle lines.

import sys, os, cv2, datetime, subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QComboBox
)
from PySide6.QtGui import QPixmap, QImage, QTextCursor, QAction
from PySide6.QtCore import Qt

# === Folders ===
script_dir = os.path.dirname(os.path.abspath(__file__))

output_folders = {
    "wide": os.path.join(script_dir, "../images/output_images/interocular_width/37_or_more"),
    "mid": os.path.join(script_dir, "../images/output_images/interocular_width/31_to_36"),
    "narrow": os.path.join(script_dir, "../images/output_images/interocular_width/30_or_less"),
}

for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Ensure symlink from cropper output to shoulder-tilt input
cropper_output = os.path.normpath(os.path.join(script_dir, "..", "images", "output_images", "cropper"))
# interocular_width_input = os.path.normpath(os.path.join(script_dir, "..", "images", "input_images", "interocular_width"))
eye_zoom = os.path.normpath(os.path.join(script_dir, "..", "images", "input_images", "eye_zoom"))

if not os.path.exists (eye_zoom):

    # run eye_zoom.py first to create the folder
    subprocess.run(["python", os.path.join(script_dir, "eye_zoom.py")])

path = interocular_width_input = eye_zoom

progress_file = os.path.join(script_dir, "../progress/interocular_width.txt")
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
        self.setWindowTitle("Interocular Width Dashboard")
        self.resize(1200, 800)

        # Log
        self.log_file = open(os.path.join(script_dir, "../session_log/interocular_width.txt"), "a", encoding="utf-8")
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
        self.files = [os.path.join(interocular_width_input, f) for f in sorted(os.listdir(interocular_width_input)) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.index = 0
        self.last_saved_path = None

        # Resume
        if os.path.exists(progress_file):
            last = open(progress_file).read().strip()
            base_names = [os.path.basename(p) for p in self.files]
            if last in base_names:
                self.index = min(base_names.index(last) + 1, len(self.files) - 1)

        if self.files:
            print(self.files[self.index])
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

        self.add_log(
            f"🖼️ Processing: {os.path.basename(filename)}, {self.index+1}-th of {len(self.files)} files "
        )

        # preview = draw_grid(img.copy())
        #self.proc_label.setPixmap(cvimg_to_qpix(preview).scaled(self.proc_label.width(), self.proc_label.height(), Qt.KeepAspectRatio))

        # instruction
        legend_x, legend_y = 0,0
        preview = img.copy()
        clue = "Click outer edge of medial corner of left eye!"
        cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
      
        self.add_log(clue)

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
                cv2.circle(img_copy, (cx, cy), 5, (0,255,0), -1)  # small dot
                cv2.putText(img_copy, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(img_copy, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)
            if i == 1:
                cv2.circle(img_copy, (cx, cy), 5, (0,255,0), -1)  # small dot
                cv2.putText(img_copy, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(img_copy, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)
            if i == 2:
                cv2.circle(img_copy, (cx, cy), 5, (200, 150,100), -1)  # small dot
                cv2.circle(img_copy, (cx, cy+40), 5, (255, 0,0), -1)  # small dot
                cv2.putText(img_copy, f"{i+1}", (cx-5, cy+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(img_copy, f"{i+1}", (cx-5, cy+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)

                # Draw vertical separator lines at line start
                p3_shifted = (cx, cy + 20)
                line_length = 40
                cv2.line(img_copy, (p3_shifted[0], p3_shifted[1] - line_length),     
                    (p3_shifted[0], p3_shifted[1]), (255, 255, 255), 1)


            if i == 3:
                cv2.circle(img_copy, (cx, cy), 5, (200, 150, 100), -1) # small dot
                cv2.circle(img_copy, (cx, cy+40), 5, (255, 0, 0), -1) # small dot
                cv2.putText(img_copy, f"{i+1}", (cx-5, cy+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(img_copy, f"{i+1}", (cx-5, cy+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)

                # Draw vertical separator lines at line end
                p4_shifted = (cx, cy + 20)
                line_length = 40
                cv2.line(img_copy, (p4_shifted[0], p4_shifted[1] - line_length),     
                    (p4_shifted[0], p4_shifted[1]), (255, 255, 255), 1)

        if step == 1:

            self.add_log("✅ Outer edge of medial corner of left eye recorded.")

            # guide line
            cv2.line(img_copy,(cx-400, cy), (cx+400, cy), (150,150,150), 1)

            # instruction
            legend_x, legend_y = 0,0
            preview = img_copy
            clue = "Click outer edge of medial corner of right eye"
            cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            self.add_log(f"2️⃣: {clue}")

        if step >= 2:

            self.add_log("✅ Outer edge of medial corner of right eye recorded.")

            self.add_log("Draw interocular line between both medial eye corners")

            p1,p2 = coords[0], coords[1]

            cv2.line(img_copy, p1, p2, (0,255,0), 2)

            lbl = "Interocular line"
            cv2.putText(img_copy, lbl, (p1[0]-80, p1[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img_copy, lbl, (p1[0]-80, p1[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            if step == 2:
                # instruction
                legend_x, legend_y = 0,0
                preview = img_copy

                clue = "Click outer edge of lateral corner of left eye!"
                cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)               


            self.add_log("3️⃣: {clue}")


        if step >= 3:
            self.add_log("✅ Outer edge of lateral corner of left eye recorded.")

            if step == 3:

                # instruction
                legend_x, legend_y = 0,0
                preview = img_copy

                clue = "Click outer edge of lateral corner of right eye!"
                cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)               

                self.add_log(f"4️⃣: {clue}")

        line_length = 100

        if step >= 4:
            self.add_log("✅ Outer edge of lateral corner of right eye recorded.")
            self.add_log("Draw line between both lateral eye corners")
            p3,p4 = coords[2], coords[3]
            cv2.line(img_copy, (p3[0], p3[1]+40), (p4[0], p4[1]+40), (255,0,0), 2)
            lbl = "Extraocular line"
            cv2.putText(img_copy, lbl, (p3[0], p3[1]+60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img_copy, lbl, (p3[0], p3[1]+60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Divide second line into 3 parts
            dx = ( p4[0] - p3[0] ) / 3.0
            dy = ( p4[1] - p3[1] ) / 3.0

            for i in range(1, 3):  # division points at 1/3 ... 2/3
                div = (int(p3_shifted[0] + i * dx), int(p3_shifted[1] + i * dy))

                # Draw vertical separator lines
                cv2.line(img_copy, (div[0], div[1] - line_length), (div[0], div[1]+20), (0, 255, 255), 1)

                cv2.circle(img_copy, (div[0], div[1]+20), 5, (200, 150, 100), -1) # small dot

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

        # numbering click order
        for i, (x, y) in enumerate(self.clicks):
            cx, cy = int(x * scale_x), int(y * scale_y)
            if i==0:
                cv2.putText(img, f"{i+1}", (cx-15, cy+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(img, f"{i+1}", (cx-15, cy+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)
            if i==1:
                cv2.putText(img, f"{i+1}", (cx+5, cy+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(img, f"{i+1}", (cx+5, cy+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)
            if i==2:
                cv2.putText(img, f"{i+1}", (cx-20, cy+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(img, f"{i+1}", (cx-20, cy+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)
            if i==3:
                cv2.putText(img, f"{i+1}", (cx+10, cy+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(img, f"{i+1}", (cx+10, cy+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)

        p1, p2, p3, p4 = [to_coords(p) for p in self.clicks]

        # First line
        cv2.line(img, p1, p2, (0, 255, 0), 2)

        # Draw small circles at first-line start
        cv2.circle(img, p1, 5, (0, 255, 0), -1)

        # Draw small circles at first-line end
        cv2.circle(img, p2, 5, (0, 255, 0), -1)

        # Labelling first line
        cv2.putText(img, "Interocular line", (p1[0]-80, p1[1]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(img, "Interocular line", (p1[0]-80, p1[1]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Shift second line 20 pixels downward
        p3_shifted = (p3[0], p3[1] + 40)
        p4_shifted = (p4[0], p4[1] + 40)

        # Second line
        cv2.line(img, p3_shifted, p4_shifted, (255, 0, 0), 2)

        # Draw small circles at start points
        cv2.circle(img, p3_shifted, 3, (255, 0, 0), -1)

        # Add labels below first point
        cv2.putText(img, f"0", (p3_shifted[0] - 5, p3_shifted[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 3)
        cv2.putText(img, f"0", (p3_shifted[0] - 5, p3_shifted[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2)

        # Draw small circles at end points
        cv2.circle(img, p4_shifted, 3, (255, 0, 0), -1)

        # Add labels below last point
        cv2.putText(img, f"1", (p4_shifted[0] - 5, p4_shifted[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 3)
        cv2.putText(img, f"1", (p4_shifted[0] - 5, p4_shifted[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2)

        # labelling second line
        cv2.putText(img, "Inter-eyetail line", (p3[0]+20, p3[1] + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(img, "Inter-eyetail line", (p3[0]+20, p3[1] + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Draw vertical separator lines
        
        line_length = 100

        # Draw vertical separator lines at line start
        cv2.line(img, (p3_shifted[0], p3_shifted[1] - line_length), (p3_shifted[0], p3_shifted[1]), (255, 255, 255), 1)

        # Draw vertical separator lines at line end
        cv2.line(img, (p4_shifted[0], p4_shifted[1] - line_length), (p4_shifted[0], p4_shifted[1]), (255, 255, 255), 1)

        # Divide second line into 3 parts
        dx = ( p4[0] - p3[0] ) / 3.0
        dy = ( p4[1] - p3[1] ) / 3.0
        
        for i in range(1, 3):  # division points at 1/3 ... 2/3
            # div = (int(p3[0] + i * dx), int(p3[1] + i * dy))
            div = (int(p3_shifted[0] + i * dx), int(p3_shifted[1] + i * dy))

            # Draw vertical separator lines
            cv2.line(img, (div[0], div[1] - line_length), (div[0], div[1]), (255, 255, 255), 1)

            # Draw small circles at division points
            cv2.circle(img, div, 3, (255, 0, 0), -1)

            # Add labels below the division points
            cv2.putText(img, f"{i}/3", (div[0] - 5, div[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 3)
            cv2.putText(img, f"{i}/3", (div[0] - 5, div[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2)

        L1 = euclidean(p1, p2)
        L2 = euclidean(p3, p4)

        # Find the shorter and longer line
        shorter = min(L1,L2)
        longer = max(L1,L2)

        # Calculate 1/3 of the longer line
        one_third = (1/3) * longer

        # Define tolerance (1% above and below 1/3)
        lower_bound = one_third * (1 - 0.10)  # 10% below 1/3
        upper_bound = one_third * (1 + 0.10)  # 10% above 1/3

        classification = "N/A" 
        # Check if shorter line is within the tolerance range around 1/3
        # if lower_bound <= shorter <= upper_bound or math.isclose(shorter, one_third, rel_tol=0.01):
        if lower_bound <= shorter <= upper_bound:
            ratio = shorter / longer
            print(f"Ratio is: {ratio}")
            classification = "Mid (31-35)"
            folder = output_folders["mid"]
        elif shorter < lower_bound :
            ratio = shorter / longer    
            print(f"Ratio is: {ratio}")
            classification = "Narrow (30 or less)"
            folder = output_folders["narrow"]
        elif shorter > upper_bound :
            ratio = shorter / longer
            print(f"Ratio is: {ratio}")
            classification ="Wide (36 or more)"
            folder = output_folders["wide"]
        else:
            print(f"The shorter line must be between {lower_bound:.3f} and {upper_bound:.3f} the length of the longer line.")
            return

        #ratio = ( shorter / longer ) if L1 else 0

        # --- Perspective classification ---
        perspective = classify_perspective(p1, p2)

        # Update combo box automatically
        index_map = {"Unlabeled": 0, "Front": 1, "Diagonal": 2, "Side": 3}
        self.view_selector.setCurrentIndex(index_map[perspective])
        self.add_log(f"🧭 Auto-perspective : {perspective}, 👁️ Please compare with your own visual judgment!")

        cv2.putText(img, f"Ratio: {ratio:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        self.add_log(f"📏 L1: {L1:.2f}, L2: {L2:.2f}, Ratio: {ratio:.2f}")

        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Legend box
        legend_x, legend_y = 0, 0
        cv2.rectangle(img, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+110), (0,0,0), -1)
        cv2.rectangle(img, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+110), (255,255,255), 1)

        cv2.putText(img, "Legend:", (legend_x, legend_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, "Green = Bimedial line", (legend_x, legend_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        # cv2.putText(img, "Yellow = Thirds", (legend_x, legend_y+35),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        cv2.putText(img, "Bilateral line", (legend_x, legend_y+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Classification + ratio + timestamp
        cv2.putText(img, f"Result: {classification}", (legend_x, legend_y+55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, f"Ratio: {ratio:.2f}", (legend_x, legend_y+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, f"Time: {timestamp}", (legend_x, legend_y+95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)

        # Save result
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

    def closeEvent(self, event):
        self.log_file.close()
        super().closeEvent(event)


# === Main ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec())