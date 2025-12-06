import os
import cv2
import mediapipe as mp
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "../images/raw_images")
output_folder = os.path.join(script_dir, "../images/input_images/eye_zoom")
session_log = os.path.join(script_dir, "../session_log/eye_zoom.txt")
os.makedirs(input_folder, exist_ok=True)

class Dashboard:
    def __init__(self,
                 input_folder=input_folder,
                 output_folder=output_folder,
                 target_width=512,
                 padding_color=(255, 255, 255)):

        # Log file
        os.makedirs(os.path.join(script_dir, "../session_log"), exist_ok=True)
        self.log_file = open(session_log, "a", encoding="utf-8")

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_width = target_width
        self.padding_color = padding_color
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )

    # logging
    def add_log(self, text):

        # console output
        print(text)

        # Still append to the session log file in ascending order
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    def crop_eyes(self, img, w, h, landmarks):
        # Get the left and right eye landmarks (indices based on the FaceMesh model)
        left_eye = landmarks.landmark[33]  # Left eye
        right_eye = landmarks.landmark[263]  # Right eye

        # Convert the normalized coordinates to pixel coordinates
        left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
        right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

        # Define a rectangle for cropping from left eye to right eye
        x1, y1 = left_eye_coords
        x2, y2 = right_eye_coords

        # Add some padding around the eyes if needed
        padding = 10
        x1 = max(x1 - padding, 0)
        x2 = min(x2 + padding, w)
        y1 = max(y1 - padding, 0)
        y2 = min(y2 + padding, h)

        # Check if the crop region is valid (not out of bounds or empty)
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid crop region for eyes: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return None

        # Crop the image around the eyes
        return img[y1:y2, x1:x2]

    """
    def resize_with_width(self, image):
        if image is None or image.size == 0:
            return None

        target_w = self.target_width
        h, w = image.shape[:2]
        scale = target_w / w
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    """

    def resize_with_padding(self, image, target_width=512, target_height=512):
        # Get original image dimensions
        h, w = image.shape[:2]
    
        # Scale image to target width while maintaining aspect ratio
        scale = target_width / w
        new_w = target_width
        new_h = int(h * scale)
    
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
        # Calculate padding for top and bottom
        top_padding = (target_height - new_h) // 2
        bottom_padding = target_height - new_h - top_padding
    
        # Add padding (top, bottom, left, right)
        padded_image = cv2.copyMakeBorder(
            resized_image,
            top=top_padding,
            bottom=bottom_padding,
            left=0,  # No left padding
            right=0,  # No right padding
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)  # White padding
        )
    
        return padded_image
    

    def process(self):
        os.makedirs(self.output_folder, exist_ok=True)
        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for filename in tqdm(image_files, desc="Processing images"):
            input_path = os.path.join(self.input_folder, filename)
            output_path = os.path.join(self.output_folder, filename)

            # Skip if already processed
            if os.path.exists(output_path):
                continue

            self.add_log("__________")
            self.add_log(f"🧾 Raw image: {input_path}")

            img = cv2.imread(input_path)
            if img is None:
                continue
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    # Crop the eye region
                    cropped_eyes = self.crop_eyes(img, w, h, landmarks)

                    if cropped_eyes is not None:
                        # Resize the cropped eye area to have width 512
                        final_img = self.resize_with_padding(cropped_eyes)

                        if final_img is not None:
                            # Save the final image
                            cv2.imwrite(output_path, final_img)
                            self.add_log(f"🖼️ Image result: {output_path}")
                        else:
                            self.add_log(f"Skipping image {filename}: Invalid crop or resize.")
                    else:
                        self.add_log(f"Skipping image {filename}: Invalid eye crop.")

            else:
                print(f"No face detected in {filename}. Skipping.")

# Run the Dashboard (for example)
if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.process()

