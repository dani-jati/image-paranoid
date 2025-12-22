import os
import cv2
import mediapipe as mp
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

input_folder = os.path.normpath(os.path.join(script_dir, "../images/raw_images"))
output_folder = os.path.normpath(os.path.join(script_dir, "../images/input_images/neck_zoom"))
session_log = os.path.normpath(os.path.join(script_dir, "../session_log/neck_zoom.txt"))

class Dashboard:
    def __init__(self,
                 input_folder=input_folder,
                 output_folder=output_folder,
                 target_size=512,
                 padding_color=(255, 255, 255)):

        os.makedirs(os.path.dirname(session_log), exist_ok=True)
        self.log_file = open(session_log, "a", encoding="utf-8")

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_size = target_size
        self.padding_color = padding_color
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5 # Slightly lowered to better handle robust features
        )

    def add_log(self, text):
        print(text)
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    def crop_neck_area(self, img, w, h, landmarks):
        
        # Crops neck area specifically for Homo javanicus morphology.
        # Landmark upper lip (Landmark 0) is needed to capture the full head-neck junction. 
        
        # MediaPipe Landmark Indices
        upper_lip = landmarks.landmark[0]      # Top center of upper lip
        face_left = landmarks.landmark[234]    # Broadest point left
        face_right = landmarks.landmark[454]   # Broadest point right

        # Horizontal bounds: Ensuring width accounts for robust jaw structure
        x1 = int(face_left.x * w)
        x2 = int(face_right.x * w)

        # Vertical bounds:
        # y1 starts exactly at the upper lip (the connection anchor)
        y1 = int(upper_lip.y * h)
        
        # Projection: We use the width of the face to determine the downward depth
        # This is often more reliable than lip-to-chin distance for robust lineages
        head_width = x2 - x1
        y2 = int(y1 + (head_width * 1.2)) # Extends down to the shoulder line

        # Final padding adjustments
        padding_x = 50 
        x1 = max(x1 - padding_x, 0)
        x2 = min(x2 + padding_x, w)
        y2 = min(y2, h)

        if x1 >= x2 or y1 >= y2:
            return None

        return img[y1:y2, x1:x2]
    
    def resize_with_padding(self, image, target_size=512):
        # (Same logic as before, ensuring 512x512 output)
        ih, iw = image.shape[:2]
        scaling_factor = min(target_size / iw, target_size / ih)
        nw, nh = int(iw * scaling_factor), int(ih * scaling_factor)
        
        resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        
        tp = (target_size - nh) // 2
        bp = target_size - nh - tp
        lp = (target_size - nw) // 2
        rp = target_size - nw - lp
    
        return cv2.copyMakeBorder(resized_image, tp, bp, lp, rp, 
                                  cv2.BORDER_CONSTANT, value=self.padding_color)

    def process(self):
        os.makedirs(self.output_folder, exist_ok=True)
        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(image_files, desc="Analyzing Neck"):
            input_path = os.path.join(self.input_folder, filename)
            output_path = os.path.join(self.output_folder, filename)
            if os.path.exists(output_path): continue

            img = cv2.imread(input_path)
            if img is None: continue
            
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                cropped_neck = self.crop_neck_area(img, w, h, landmarks)

                if cropped_neck is not None:
                    final_img = self.resize_with_padding(cropped_neck, self.target_size)
                    cv2.imwrite(output_path, final_img)
                    self.add_log(f"✔️ Captured: {filename}")
            else:
                self.add_log(f"❓ Structure not recognized: {filename}")

if __name__ == "__main__":
    app = Dashboard()
    app.process()
