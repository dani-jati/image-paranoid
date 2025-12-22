# this script is to measure mouth junction shape.
# mouth junction shape is the geometric configuration of the upper and lower lip junction, including concave, linear, and convex forms.


import sys, os, cv2, datetime, subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QComboBox
)
from PySide6.QtGui import QPixmap, QImage, QTextCursor, QAction
from PySide6.QtCore import Qt
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# mode importing module: as main or as module.
if __name__ == "__main__":
    from line import Line
else:
    from tools.line import Line

# === Folders ===
script_dir = os.path.dirname(os.path.abspath(__file__))

cropper_output = os.path.join(script_dir, "../images/output_images/cropper")
mouth_zoom = os.path.normpath(os.path.join(script_dir, "..", "images", "input_images", "mouth_zoom"))

# Check for folder input
if not os.path.exists(mouth_zoom):

    # run mouth_zoom.py first to create the input folder "mouth_zoom".
    subprocess.run(["python", os.path.join(script_dir, "mouth_zoom.py")])


path = mouthjunction_path_input = mouth_zoom

# print("Symlink points to:", os.readlink(mouthjunction_path_input))

output_folders = {
    "concave": os.path.join(script_dir, "../images/output_images/mouthjunction_path/concave"),
    "linier": os.path.join(script_dir, "../images/output_images/mouthjunction_path/linier"),
    "convex": os.path.join(script_dir, "../images/output_images/mouthjunction_path/convex"),
}
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

progress_file = os.path.join(script_dir, "../progress/mouthjunction_path.txt")
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

def label_guide_lines(img, coord):
    cv2.putText(img, f"v", coord,
            cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 3)
    cv2.putText(img, f"v", coord,
            cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


# === Dashboard ===
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mouth Junction Path Dashboard")
        self.resize(1200, 800)

        # Log
        self.log_file = open(os.path.join(script_dir, "../session_log/mouthjunction_path.txt"), "a", encoding="utf-8")
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
        self.files = [os.path.join(mouthjunction_path_input, f) for f in sorted(os.listdir(mouthjunction_path_input)) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
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

        self.add_log(
            f"🖼️ Processing: {os.path.basename(filename)}, {self.index+1}-th of {len(self.files)} files "
        )

        # instruction
        legend_x, legend_y = 0,0
        preview = img.copy()
        clue = "Click the leftmost point of mouth junction!"
        cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 3)
        cv2.putText(preview, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,0), 2)

        self.add_log(" 1️⃣: " + clue)
        
        self.proc_label.setPixmap(
            cvimg_to_qpix(preview).scaled(
                self.proc_label.width(), 
                self.proc_label.height(), 
                Qt.KeepAspectRatio
            )
        )

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

            cv2.circle(img_copy, (cx, cy), 5, (0, 255, 0), -1)

            cv2.putText(img_copy, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 3)
            cv2.putText(img_copy, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (90,255,255), 2)

        if step == 1:
            self.add_log("✅ The leftmost point of mouth junction recorded.")

            # instruction
            legend_x, legend_y = 0,0
            clue = "Click the rightmost point of mouth junction!"
            cv2.putText(img_copy, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 3)
            cv2.putText(img_copy, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
            
            self.add_log("2️⃣: {clue}")

        elif step >= 2 :
            self.add_log("the rightmost point of mouth junction recorded")

            self.add_log("draw line from p1 to p2")
            p1,p2 = coords[0], coords[1]

            # guide line
            cv2.line(img_copy, (p1[0], p1[1]+40), (p2[0],p1[1]+40), (255,0,0), 3)

            cv2.circle(img_copy, (p1[0], p1[1]+40), 5, (255, 0, 0), -1)
            cv2.circle(img_copy, (p2[0], p1[1]+40), 5, (255, 0, 0), -1)

            # split guide line to 2 equal parts
            dtx = ( p2[0] - p1[0] ) / 2.0
            dty = ( p2[1] - p1[1] ) / 2.0

            p1_shifted = (p1[0], p1[1]+40)
            line_length = 70

            dit = (int(p1_shifted[0] + 1 * dtx), int(p1_shifted[1] + 1 * dty))
            cv2.line(img_copy, (dit[0], dit[1] - line_length), (dit[0], dit[1] + 0), (100, 100, 100, 50), 2)

            # instruction
            legend_x, legend_y = 0,0
            clue = "Click a junction point with upper lip path under arrow v"
            cv2.putText(img_copy, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 3)
            cv2.putText(img_copy, clue, (legend_x+10, legend_y+30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

            cv2.line(img_copy, (dit[0], 100 ), (dit[0], dit[1] ), (150, 150, 150), 1)

            if step == 2 :
                cv2.putText(img_copy, f"v", (dit[0]-4, 90 ),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

        if step >= 3 :

            # Divide second line into 7 parts
            dx = ( p2[0] - p1[0] ) / 7.0
            dy = ( p2[1] - p1[1] ) / 7.0

            for i in range(0, 8):  # division points at 1/7 ... 2/7

                # determine x coordinate
                div = (int(p1_shifted[0] + i * dx), int(p1_shifted[1] + i * dy))

                # draw point
                # cv2.circle(img_copy, (div[0], div[1] + 0), 5, (200, 150, 100), -1) # small dot

                # grey line from top to division point
                overlay = img_copy.copy()
                alpha = 0.5
                cv2.line(overlay, (div[0], 100 ), (div[0], div[1] ), (200, 200, 200), 1)
                img_copy = cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha,  0)

                # if step is always i + 2
                if step == i + 2:
                    label_guide_lines(img_copy, (div[0] - 4, 90))
                    
        coords.sort()
        self.coords = coords

        if len(coords) == 9 :
            img_copy = self.raw_img.copy()

            # draw all points so far
            for i, (x, y) in enumerate(coords):
                cx, cy = int(x * scale_x), int(y * scale_y)
                cv2.circle(img_copy, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(img_copy, f"{i+1}", (cx-5, cy+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(img_copy, f"{i+1}", (cx-5, cy+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)

            pts = np.array(coords, np.int32)

            # Reshape the array to the required shape (must be a 2D array)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw the polyline on the image (True to close the polyline, False to leave it open)
            cv2.polylines(img_copy, [pts], isClosed=False, color=(0, 0, 255), thickness=1)

            x = np.array([coord[0] for coord in coords])
            y = np.array([coord[1] for coord in coords])

            # Reshape data for sklearn
            x_reshaped = x.reshape(-1, 1)

            # Fit a linear model
            model = LinearRegression()
            model.fit(x_reshaped, y)

            # Predict y values
            y_pred = model.predict(x_reshaped)

            # Calculate residuals (difference between actual and predicted y-values)
            residuals = y - y_pred
            sse = np.sum(residuals**2)  # Sum of squared residuals

            # Evaluate the fit
            print(f"Sum of squared residuals: {sse}")

            # Set a threshold (e.g., 5)
            threshold = 1000

            # Check if SSE is below the threshold
            is_straight = None
            if sse < threshold:
                self.is_straight = is_straight = True
                self.curvature = "linier"
                print("Is the lip-junction path circa straight? True")
            else:
                self.is_straight = is_straight = False
                print("Is the lip-junction path circa straight? False")

                # Step 1: Fit a quadratic curve to the data
                coefficients = np.polyfit(x, y, 2)  # Fit a second-degree polynomial (quadratic)
                a, b, c = coefficients  # Extract the coefficients

                # Step 2: Create the fitted quadratic curve
                fitted_y = np.polyval(coefficients, x)
    
                # Step 3: Plot the original data and the quadratic fit
                # plt.plot(x, y, 'o', label="Original Data")
                # plt.plot(x, fitted_y, '-', label=f"Fitted Quadratic: $f(x) = {a:.2f}x^2 + {b:.2f}x + {c:.2f}$")
                # plt.title("Quadratic Fit to Data")
                # plt.xlabel("X")
                # plt.ylabel("Y")
                # plt.legend()
                # plt.grid(True)
                # plt.show()

                # Step 4: Analyze the sign of 'a' to determine concavity
                if a < 0:
                    print("The lip junction is concave up.")
                    self.curvature = "concave"
                elif a > 0:
                    print("The lip junction is concave down.")
                    self.curvature = "convex"
                else:
                    print("The curve is linear (no concavity).")

            # Plot the data and the fitted line
            # plt.scatter(x, y, label='Data points')
            # plt.plot(x, y_pred, color='red', label='Fitted line')
            # plt.legend()
            # plt.show()

        # update preview
        self.proc_label.setPixmap(
            cvimg_to_qpix(img_copy).scaled(
                self.proc_label.width(), self.proc_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        
        if len(self.clicks) == 9:
            self.process_clicks()
        

    def process_clicks(self):

        print( "process_clicks" )

        img = self.raw_img.copy()
        h, w = img.shape[:2]
        disp_w, disp_h = self.proc_label.width(), self.proc_label.height()
        scale_x, scale_y = w / disp_w, h / disp_h

        def to_coords(p):  # p is a tuple (x, y)
            return int(p[0] * scale_x), int(p[1] * scale_y)

        # draw all points so far
        for i, (x, y) in enumerate(self.coords):
            cx, cy = int(x * scale_x), int(y * scale_y)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(img, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img, f"{i+1}", (cx-5, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)

        pts = np.array(self.coords, np.int32)

        # Reshape the array to the required shape (must be a 2D array)
        pts = pts.reshape((-1, 1, 2))
            
        # Draw the polyline on the image (True to close the polyline, False to leave it open)
        cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=1)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Legend box
        legend_x, legend_y = 0, 0
        cv2.rectangle(img, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+115), (0,0,0), -1)
        cv2.rectangle(img, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+115), (255,255,255), 1)

        cv2.putText(img, "Legend:", (legend_x, legend_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, "Green = Mouth junction path", (legend_x, legend_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Classification + ratio + timestamp
        cv2.putText(img, f"Result:", (legend_x, legend_y+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, f"  Is straight: {'Yes' if self.is_straight == True else 'No'}", (legend_x, legend_y+55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, f"  Curvature: {self.curvature}", (legend_x, legend_y+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        cv2.putText(img, f"Time: {timestamp}", (legend_x, legend_y+95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)


        # update preview
        self.result_label.setPixmap(cvimg_to_qpix(img).scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))

        folder = output_folders[ self.curvature ]

        # save result
        out_path = os.path.join(folder, os.path.basename(self.filename))
        cv2.imwrite(out_path, img)
        self.last_saved_path = out_path
        self.add_log(f"✅ Saved to {out_path}")

        # Logging progress
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

    def closeEvent(self, event):
        self.log_file.close()
        super().closeEvent(event)
    

# === Main ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec())


