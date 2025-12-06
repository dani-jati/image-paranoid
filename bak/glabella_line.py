import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# === CONFIGURATION ===
source_folder = "source"   # Folder with input images
target_folder = "target"   # Folder to save processed images
N = 5                      # Number of frames for smoothing (set to 1 for no smoothing)
LEFT_INNER = 336           # Landmark indices for inner eyebrow (approx glabella)
RIGHT_INNER = 107

# === PREPARE OUTPUT FOLDER ===
os.makedirs(target_folder, exist_ok=True)

# === Initialize MediaPipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # important for still images
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# === For smoothing glabella positions across images ===
glabella_history = deque(maxlen=N)

# === Process all images in source folder ===
for filename in sorted(os.listdir(source_folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue  # skip non-image files

    image_path = os.path.join(source_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not read {filename}")
        continue

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # Get pixel coords of inner eyebrow landmarks
        x1, y1 = int(lm[LEFT_INNER].x * w), int(lm[LEFT_INNER].y * h)
        x2, y2 = int(lm[RIGHT_INNER].x * w), int(lm[RIGHT_INNER].y * h)

        # Midpoint between eyebrows
        glab_x = (x1 + x2) // 2
        glab_y = (y1 + y2) // 2

        # Shift glabella point slightly downward by 2.05% of image height
        delta_y = int(0.0205 * h)
        glab_y = min(h - 1, glab_y + delta_y)

        # Add to smoothing buffer
        glabella_history.append([glab_x, glab_y])
        smoothed_glabella = np.mean(glabella_history, axis=0).astype(int)
        gx, gy = smoothed_glabella

        # Draw green circle on glabella
        cv2.circle(image, (gx, gy), 5, (0, 255, 0), -1)

        # Draw horizontal red line across the entire image at glabella height
        cv2.line(image, (0, gy), (w, gy), (0, 0, 255), 2)

        print(f"✅ Processed {filename} — glabella at ({gx}, {gy})")
    else:
        print(f"❌ No face detected in {filename}")

    # Save the processed image to target folder
    out_path = os.path.join(target_folder, filename)
    cv2.imwrite(out_path, image)

print("✅ All images processed and saved to:", os.path.abspath(target_folder))







