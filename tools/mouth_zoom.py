import os
import cv2
import mediapipe as mp
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

input_folder = os.path.normpath(os.path.join(script_dir, "../images/raw_images"))
output_folder = os.path.normpath(os.path.join(script_dir, "../images/input_images/mouth_zoom"))
session_log = os.path.normpath(os.path.join(script_dir, "../session_log/mouth_zoom.txt"))

class Dashboard:
    def __init__(self,
                 input_folder=input_folder,
                 output_folder=output_folder,
                 target_size=512, # Menggunakan target_size agar simetris
                 padding_color=(255, 255, 255)):

        os.makedirs(os.path.dirname(session_log), exist_ok=True)
        self.log_file = open(session_log, "a", encoding="utf-8")

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_size = target_size
        self.padding_color = padding_color
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, # Optimalize for image processing (not video)
            max_num_faces=1,
            min_detection_confidence=0.6
        )

    def add_log(self, text):
        print(text)
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    def crop_lower_face(self, img, w, h, landmarks):
        # MediaPipe Landmark Indices
        nose_tip = landmarks.landmark[164] # nose-mouth border
        chin = landmarks.landmark[200] # mouth-chin border          
        mouth_left = landmarks.landmark[61]     
        mouth_right = landmarks.landmark[291]   

        x1 = int(mouth_left.x * w)
        x2 = int(mouth_right.x * w)
        y1 = int(nose_tip.y * h)
        y2 = int(chin.y * h)

        # Padding area crop (dalam pixel)
        padding_x = 40 
        padding_y = 20
        
        x1 = max(x1 - padding_x, 0)
        x2 = min(x2 + padding_x, w)
        y1 = max(y1 - padding_y, 0)
        y2 = min(y2 + padding_y, h)

        if x1 >= x2 or y1 >= y2:
            return None

        return img[y1:y2, x1:x2]
    
    def resize_with_padding(self, image, target_size=512):
        h, w = image.shape[:2]
        
        # Determine ratio to resize so that image fits within a 512x512 panel
        scaling_factor = min(target_size / w, target_size / h)
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calculate padding to make image square (512x512)
        top_padding = (target_size - new_h) // 2
        bottom_padding = target_size - new_h - top_padding
        left_padding = (target_size - new_w) // 2
        right_padding = target_size - new_w - left_padding
    
        padded_image = cv2.copyMakeBorder(
            resized_image,
            top=top_padding,
            bottom=bottom_padding,
            left=left_padding,
            right=right_padding,
            borderType=cv2.BORDER_CONSTANT,
            value=self.padding_color
        )
    
        return padded_image

    def process(self):
        os.makedirs(self.output_folder, exist_ok=True)
        if not os.path.exists(self.input_folder):
            self.add_log(f"❌ Folder input tidak ditemukan: {self.input_folder}")
            return

        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(image_files, desc="Processing"):
            input_path = os.path.join(self.input_folder, filename)
            output_path = os.path.join(self.output_folder, filename)

            if os.path.exists(output_path):
                continue

            img = cv2.imread(input_path)
            if img is None: continue
            
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                # Ambil wajah pertama yang terdeteksi
                landmarks = results.multi_face_landmarks[0]
                cropped_mouth = self.crop_lower_face(img, w, h, landmarks)

                if cropped_mouth is not None:
                    final_img = self.resize_with_padding(cropped_mouth, self.target_size)
                    cv2.imwrite(output_path, final_img)
                    self.add_log(f"✔️ Berhasil: {filename}")
                else:
                    self.add_log(f"⚠️ Gagal crop: {filename}")
            else:
                self.add_log(f"❓ Wajah tidak ditemukan: {filename}")

if __name__ == "__main__":
    app = Dashboard()
    app.process()