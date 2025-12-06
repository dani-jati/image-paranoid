import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# === CONFIGURATION ===
source_folder = "cropped_imgs/Ridwan Kamil/frontal"    # Folder with input images
target_folder = "facial_thirds_imgs/Ridwan Kamil/frontal"    # Folder to save processed images
N = 5                       # Frames for smoothing glabella (set 1 for no smoothing)

# --- Landmark indices ---
LEFT_INNER = 336            # Inner left eyebrow (approx glabella)
RIGHT_INNER = 107           # Inner right eyebrow
LEFT_ALA = 94               # Left alar base
RIGHT_ALA = 323             # Right alar base
CHIN_TIP = 152              # Lowest point of chin
PHILTRUM_OFFSET_Y = 10      # Offset downward from mid-alar line (in pixels)

# === PREPARE OUTPUT FOLDER ===
os.makedirs(target_folder, exist_ok=True)

# === Initialize MediaPipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# === History buffer for smoothing glabella position ===
glabella_history = deque(maxlen=N)

# === Process all images ===
for filename in sorted(os.listdir(source_folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(source_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not read {filename}")
        continue

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print(f"❌ No face detected in {filename}")
        continue

    lm = results.multi_face_landmarks[0].landmark

    # === Glabela calculation ===
    x1, y1 = int(lm[LEFT_INNER].x * w), int(lm[LEFT_INNER].y * h)
    x2, y2 = int(lm[RIGHT_INNER].x * w), int(lm[RIGHT_INNER].y * h)
    glab_x = (x1 + x2) // 2
    glab_y = (y1 + y2) // 2
    delta_y = int(0.0205 * h)
    glab_y = min(h - 1, glab_y + delta_y)
    glabella_history.append([glab_x, glab_y])
    gx, gy = np.mean(glabella_history, axis=0).astype(int)

    # === Phyltrum calculation ===
    left_ala_x, left_ala_y = int(lm[LEFT_ALA].x * w), int(lm[LEFT_ALA].y * h)
    right_ala_x, right_ala_y = int(lm[RIGHT_ALA].x * w), int(lm[RIGHT_ALA].y * h)
    mid_x = (left_ala_x + right_ala_x) // 2
    mid_y = (left_ala_y + right_ala_y) // 2
    phyltrum_y = min(h - 1, mid_y + PHILTRUM_OFFSET_Y)

    # === Chin calculation ===
    chin_x = int(lm[CHIN_TIP].x * w)
    chin_y = int(lm[CHIN_TIP].y * h)

    # === Draw horizontal lines ===
    cv2.line(image, (0, gy), (w, gy), (0, 0, 255), 2)         # Red glabella line
    cv2.line(image, (0, phyltrum_y), (w, phyltrum_y), (0, 255, 255), 2)  # Cyan phyltrum line
    cv2.line(image, (0, chin_y), (w, chin_y), (255, 0, 0), 2) # Blue chin line

    print(f"✅ {filename}: Glabella y={gy}, Phyltrum y={phyltrum_y}, Chin y={chin_y}")

    # === Save result ===
    out_path = os.path.join(target_folder, filename)
    cv2.imwrite(out_path, image)

print("✅ All images processed and saved to:", os.path.abspath(target_folder))
