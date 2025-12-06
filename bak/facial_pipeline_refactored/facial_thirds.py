import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

def main():
    source_folder = "cropped_imgs"
    target_folder = "facial_thirds_imgs"
    N = 5
    os.makedirs(target_folder, exist_ok=True)
    LEFT_INNER, RIGHT_INNER = 336, 107
    LEFT_ALA, RIGHT_ALA, CHIN_TIP = 94, 323, 152
    PHILTRUM_OFFSET_Y = 10

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5
    )
    glabella_history = deque(maxlen=N)

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
        x1, y1 = int(lm[LEFT_INNER].x * w), int(lm[LEFT_INNER].y * h)
        x2, y2 = int(lm[RIGHT_INNER].x * w), int(lm[RIGHT_INNER].y * h)
        glab_x, glab_y = (x1 + x2) // 2, (y1 + y2) // 2
        delta_y = int(0.0205 * h)
        glab_y = min(h - 1, glab_y + delta_y)
        glabella_history.append([glab_x, glab_y])
        gx, gy = np.mean(glabella_history, axis=0).astype(int)
        left_ala_y = int(lm[LEFT_ALA].y * h)
        right_ala_y = int(lm[RIGHT_ALA].y * h)
        mid_y = (left_ala_y + right_ala_y) // 2
        phyltrum_y = min(h - 1, mid_y + PHILTRUM_OFFSET_Y)
        chin_y = int(lm[CHIN_TIP].y * h)
        cv2.line(image, (0, gy), (w, gy), (0, 0, 255), 2)
        cv2.line(image, (0, phyltrum_y), (w, phyltrum_y), (0, 255, 255), 2)
        cv2.line(image, (0, chin_y), (w, chin_y), (255, 0, 0), 2)
        print(f"✅ {filename}: Glabella y={gy}, Phyltrum y={phyltrum_y}, Chin y={chin_y}")
        cv2.imwrite(os.path.join(target_folder, filename), image)

    print("✅ All images processed and saved to:", os.path.abspath(target_folder))

if __name__ == "__main__":
    main()
