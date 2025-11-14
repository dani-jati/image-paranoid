import os
import cv2
import mediapipe as mp
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "../images/raw_images")
output_folder = os.path.join(script_dir, "../images/output_images/cropper")
session_log = os.path.join(script_dir, "../session_log/cropper.txt")
os.makedirs(input_folder, exist_ok=True)

class Dashboard:
    def __init__(self,
                 input_folder=input_folder,
                 output_folder=output_folder,
                 target_size=(512, 512),
                 padding_color=(255, 255, 255)):

        # Log file
        os.makedirs(os.path.join(script_dir, "../session_log"), exist_ok=True)
        self.log_file = open(session_log, "a", encoding="utf-8")

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_size = target_size
        self.padding_color = padding_color
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )

    # logging
    def add_log(self, text):

        # console output
        print(text)

        # Still append to the session log file in ascending order
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    def crop_head_shoulders(self, img, w, h, bbox):
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        top = max(y - int(0.5 * bh), 0)
        bottom = min(y + int(1.7 * bh), h)
        left = max(x - int(0.9 * bw), 0)
        right = min(x + int(1.9 * bw), w)
        return img[top:bottom, left:right]

    def resize_with_padding(self, image):
        target_w, target_h = self.target_size
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        padded = cv2.copyMakeBorder(
            resized,
            top=(target_h - new_h) // 2,
            bottom=(target_h - new_h + 1) // 2,
            left=(target_w - new_w) // 2,
            right=(target_w - new_w + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=self.padding_color
        )
        return padded

    def process(self):
        os.makedirs(self.output_folder, exist_ok=True)
        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for filename in tqdm(image_files, desc="Processing images"):
            input_path = os.path.join(self.input_folder, filename)
            output_path = os.path.join(self.output_folder, filename)

            # skip if already processed
            if os.path.exists (output_path):
                continue

            self.add_log("__________")
            self.add_log(f"🧾 Raw image: { input_path }")

            img = cv2.imread(input_path)
            if img is None:
                continue
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(img_rgb)
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                cropped = self.crop_head_shoulders(img, w, h, bbox)
                final_img = self.resize_with_padding(cropped)
                cv2.imwrite(output_path, final_img)
                
                self.add_log(f"🖼️ Image result: { output_path }")
            else:
                print(f"No face detected in {filename}. Skipping.")




