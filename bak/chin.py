import cv2
import mediapipe as mp
import os

# --- Configuration ---
source_folder = "source"   # Input images folder
target_folder = "target"   # Output images folder

# Chin tip landmark index (MediaPipe FaceMesh)
CHIN_TIP = 152

# Ensure output folder exists
os.makedirs(target_folder, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# Process all images in the source folder
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

        # Get pixel coordinates of chin tip (lowest point)
        chin_x = int(lm[CHIN_TIP].x * w)
        chin_y = int(lm[CHIN_TIP].y * h)

        # Draw a horizontal blue line at chin level
        cv2.line(image, (0, chin_y), (w, chin_y), (255, 0, 0), 2)

        # Optionally mark the chin point itself
        # cv2.circle(image, (chin_x, chin_y), 6, (255, 0, 0), -1)

        print(f"{filename}: Chin line drawn at y = {chin_y}")
    else:
        print(f"No face detected in {filename}")

    out_path = os.path.join(target_folder, filename)
    cv2.imwrite(out_path, image)

print("Processing complete. Results saved in:", os.path.abspath(target_folder))
