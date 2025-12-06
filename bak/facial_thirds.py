import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# ====================================================================
# === 1. CONFIGURATION AND LANDMARK DEFINITIONS (IMPROVED GLABELLA) ===
# ====================================================================
# Configuration
source_folder = "cropped_imgs/Ridwan Kamil/frontal"    # Folder with input images
target_folder = "facial_thirds_imgs/Ridwan Kamil/frontal"    # Folder to save processed images
SMOOTHING_FRAMES = 5                                  # N: Frames for smoothing (set 1 for no smoothing)

# Standard Anatomical Points for Facial Thirds
# REVISED GLABELLA ESTIMATION:
# We use the inner ends of the eyebrows for a more anatomically correct position.
LM_L_INNER_EYEBROW = 107 # Inner left eyebrow end (Approximation)
LM_R_INNER_EYEBROW = 336 # Inner right eyebrow end (Approximation)
GLABELLA_Y_OFFSET_RATIO = -0.015 # Negative offset moves the line UP from the eyebrow average.
                                  # This ratio is relative to image height (h). 
                                  # Adjust this value (-0.010 to -0.020) to fine-tune the anatomical Glabella position.

# Mid-face bottom line (Subnasale/Alar Base)
LM_R_ALA = 94        # Right Alar Base 
LM_L_ALA = 323       # Left Alar Base 

# Lowest point of chin (Menton/Pogonion)
LM_POGONION = 152    # Lowest point of the chin

# Colors (BGR)
COLOR_GLABELLA = (0, 0, 255)      # Red
COLOR_MID_FACE = (0, 255, 255)    # Cyan/Yellow
COLOR_CHIN = (255, 0, 0)          # Blue

# ====================================================================
# === 2. SETUP ===
# ====================================================================

# Prepare output folder
os.makedirs(target_folder, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# History buffer for smoothing glabella position
glabella_history = deque(maxlen=SMOOTHING_FRAMES)

# ====================================================================
# === 3. PROCESSING LOOP ===
# ====================================================================

print(f"Starting facial landmark processing on {source_folder}...")

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
        glabella_history.clear()
        continue

    lm = results.multi_face_landmarks[0].landmark

    # Helper function to get pixel coordinates
    def get_coords(landmark_index):
        point = lm[landmark_index]
        return int(point.x * w), int(point.y * h)
    
    # --- A. Glabella (Upper Third Line) ---
    # 1. Get inner eyebrow points
    _, y_L_brow = get_coords(LM_L_INNER_EYEBROW)
    x_R_brow, y_R_brow = get_coords(LM_R_INNER_EYEBROW) # x_R_brow is used for drawing the point

    # 2. Calculate the base Y-coordinate from the average inner eyebrow Y
    base_y_brow = (y_L_brow + y_R_brow) // 2

    # 3. Apply the proportional offset to move the line *up* to the anatomical Glabella
    # Note: Since Y increases downwards, a NEGATIVE offset moves the line UP.
    glab_y = base_y_brow + int(GLABELLA_Y_OFFSET_RATIO * h)

    # 4. Apply smoothing
    glabella_history.append(glab_y)
    smoothed_y = np.mean(glabella_history).astype(int)
    gy = np.clip(smoothed_y, 0, h - 1)
    
    # --- B. Mid-Face Bottom (Subnasale / Alar Base Line) ---
    r_ala_x, r_ala_y = get_coords(LM_R_ALA)
    l_ala_x, l_ala_y = get_coords(LM_L_ALA)
    
    # Midpoint of Alar Bases (approximates Subnasale)
    mid_face_y = (r_ala_y + l_ala_y) // 2
    mid_face_y = np.clip(mid_face_y, 0, h - 1)
    
    # --- C. Lower Point of Chin (Menton/Pogonion) ---
    _, chin_y = get_coords(LM_POGONION)
    chin_y = np.clip(chin_y, 0, h - 1)
    
    # ====================================================================
    # === 4. DRAWING AND OUTPUT ===
    # ====================================================================

    # Draw horizontal lines (Left Edge to Right Edge)
    cv2.line(image, (0, gy), (w, gy), COLOR_GLABELLA, 2)     # Red Glabella line
    cv2.line(image, (0, mid_face_y), (w, mid_face_y), COLOR_MID_FACE, 2) # Cyan/Yellow Alar line
    cv2.line(image, (0, chin_y), (w, chin_y), COLOR_CHIN, 2) # Blue Chin line
    
    # Draw points for visual confirmation
    # Glabella point is positioned mid-way between the inner eyebrows at the calculated Y
    glab_x_center = (get_coords(LM_L_INNER_EYEBROW)[0] + x_R_brow) // 2
    cv2.circle(image, (glab_x_center, gy), 5, COLOR_GLABELLA, -1)
    cv2.circle(image, ((r_ala_x + l_ala_x) // 2, mid_face_y), 5, COLOR_MID_FACE, -1)
    cv2.circle(image, get_coords(LM_POGONION), 5, COLOR_CHIN, -1)

    print(f"✅ {filename}: Glabella y={gy}, Mid-Face Bottom y={mid_face_y}, Chin y={chin_y}")

    # Save result
    out_path = os.path.join(target_folder, filename)
    cv2.imwrite(out_path, image)

print("\n✅ All images processed and saved to:", os.path.abspath(target_folder))
