import cv2
import mediapipe as mp
import os

# --- Configuration ---
source_folder = "source"  # Your input images folder
target_folder = "target"  # Your output images folder

# Landmarks for left and right alar bases
LEFT_ALA = 94
RIGHT_ALA = 323

# Offset to move the horizontal line down (in pixels)
OFFSET_Y = 10

os.makedirs(target_folder, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

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

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # Get pixel coordinates of left and right alae
        left_ala_x, left_ala_y = int(lm[LEFT_ALA].x * w), int(lm[LEFT_ALA].y * h)
        right_ala_x, right_ala_y = int(lm[RIGHT_ALA].x * w), int(lm[RIGHT_ALA].y * h)

        # Compute midpoint
        mid_x = (left_ala_x + right_ala_x) // 2
        mid_y = (left_ala_y + right_ala_y) // 2

        # Apply downward offset to the y-coordinate
        line_y = mid_y + OFFSET_Y
        if line_y >= h:
            line_y = h - 1  # Prevent going beyond image boundary

        # Draw horizontal line across image at offset y
        cv2.line(image, (0, line_y), (w, line_y), (0, 255, 255), 2)  # cyan line

        # Mark the adjusted midpoint position
        # cv2.circle(image, (mid_x, line_y), 7, (0, 255, 255), -1)  # cyan circle

        print(f"{filename}: Line drawn at y = {line_y}")

    else:
        print(f"No face detected in {filename}")

    out_path = os.path.join(target_folder, filename)
    cv2.imwrite(out_path, image)

print("Processing complete. Results saved in:", target_folder)



