import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
import glob
import math

# --- Configuration ---
INPUT_DIR = 'input_photos'       # Directory where your source photos are located
OUTPUT_DIR_PLUS = '42_plus'      # Folder for max_tilt >= 42 degrees
OUTPUT_DIR_MINUS = '41_minus'    # Folder for max_tilt < 42 degrees
TILT_THRESHOLD = 42.0            # The angle threshold in degrees

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,       
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------------------------------------------------
# --- KEY FUNCTION: 2D 3-Point Angle Calculation for Anchor-Based Tilt ---
# ----------------------------------------------------------------------
def calculate_tilt_angle_3pts(a_coords, b_coords, c_coords):
    """
    Calculates the 2D angle (in degrees) between the line segments BA and BC.
    B: Anchor Point (e.g., Ear); A: Horizontal Reference; C: Target Point (e.g., Shoulder)
    """
    a = np.array(a_coords) 
    b = np.array(b_coords) 
    c = np.array(c_coords) 

    ba = a - b 
    bc = c - b

    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    
    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    try:
        cos_theta = dot_product / (mag_ba * mag_bc)
        cos_theta = np.clip(cos_theta, -1.0, 1.0) 
        
        radians = np.arccos(cos_theta)
        return np.degrees(radians)
        
    except Exception:
        return 0.0

# ----------------------------------------------------------------------
# --- Main Processing Function (Checks BOTH Shoulders) ---
# ----------------------------------------------------------------------
def process_photo(image_path):
    """
    Calculates the 2D tilt for BOTH shoulders (relative to the horizontal ear anchor)
    and returns the maximum tilt found.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        screen_landmarks = results.pose_landmarks.landmark
        
        left_tilt = 0.0
        right_tilt = 0.0
        
        # --- 1. Measure LEFT Shoulder Tilt ---
        l_anchor = screen_landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        l_target = screen_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        if l_anchor.visibility >= 0.5 and l_target.visibility >= 0.5:
            l_b_coords = np.array([l_anchor.x, l_anchor.y])
            l_c_coords = np.array([l_target.x, l_target.y])
            l_a_coords = np.array([l_anchor.x + 0.1, l_anchor.y]) 
            left_tilt = calculate_tilt_angle_3pts(l_a_coords, l_b_coords, l_c_coords)

        # --- 2. Measure RIGHT Shoulder Tilt ---
        r_anchor = screen_landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
        r_target = screen_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if r_anchor.visibility >= 0.5 and r_target.visibility >= 0.5:
            r_b_coords = np.array([r_anchor.x, r_anchor.y])
            r_c_coords = np.array([r_target.x, r_target.y])
            r_a_coords = np.array([r_anchor.x + 0.1, r_anchor.y])
            right_tilt = calculate_tilt_angle_3pts(r_a_coords, r_b_coords, r_c_coords)

        # --- 3. Determine the Max Tilt ---
        max_tilt = max(left_tilt, right_tilt)
        
        # If both tilts are 0.0 and one side was not visible, return None to skip
        if max_tilt == 0.0 and (l_anchor.visibility < 0.5 and r_anchor.visibility < 0.5):
            return None
        
        # --- Visualization (using the side with the maximum tilt for drawing) ---
        img_h, img_w, _ = image.shape
        
        if left_tilt >= right_tilt and left_tilt > 0:
            b_coords, c_coords = l_b_coords, l_c_coords
            side = "Left"
        elif right_tilt > 0:
            b_coords, c_coords = r_b_coords, r_c_coords
            side = "Right"
        else:
            return max_tilt
        
        p_anchor = (int(b_coords[0] * img_w), int(b_coords[1] * img_h))
        p_target = (int(c_coords[0] * img_w), int(c_coords[1] * img_h))
        p_reference = (int((b_coords[0] + 0.1) * img_w), int(b_coords[1] * img_h))
        
        cv2.line(image, p_anchor, p_target, (0, 255, 0), 2)
        cv2.line(image, p_anchor, p_reference, (255, 0, 0), 2)
        text = f"MAX Tilt: {max_tilt:.2f} deg ({side})"
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(f"annotated_{os.path.basename(image_path)}", image)
        
        return max_tilt
    else:
        return None

# ----------------------------------------------------------------------
# --- Main Sorting Function (Uses Copy instead of Move) ---
# ----------------------------------------------------------------------
def sort_photos():
    """
    Main function to set up folders, process all images, and copy them 
    based on the maximum tilt of either shoulder.
    """
    os.makedirs(OUTPUT_DIR_PLUS, exist_ok=True)
    os.makedirs(OUTPUT_DIR_MINUS, exist_ok=True)
    
    # Robust search for common image extensions (case-insensitive)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    image_paths = list(set(image_paths))
    
    if not image_paths:
        print(f"No images found in the directory: {INPUT_DIR}. Please ensure images are present.")
        return

    print(f"Found {len(image_paths)} images to process...")
    
    for i, image_path in enumerate(image_paths):
        file_name = os.path.basename(image_path)
        print(f"Processing ({i+1}/{len(image_paths)}): {file_name}")
        
        max_tilt = process_photo(image_path)
        
        if max_tilt is not None:
            if max_tilt >= TILT_THRESHOLD:
                destination_folder = OUTPUT_DIR_PLUS
            else:
                destination_folder = OUTPUT_DIR_MINUS
            
            print(f" -> MAX Tilt {max_tilt:.2f}. **Copying** to **{destination_folder}**.")
            
            try:
                # *** IMPORTANT: Using copy2() to KEEP the original image in INPUT_DIR ***
                shutil.copy2(image_path, os.path.join(destination_folder, file_name))
            except Exception as e:
                print(f"Error copying file {file_name}: {e}")
        else:
            print(f" -> Could not determine tilt for {file_name} (Landmarks not visible). Skipping copy.")

    print("\n--- Processing Complete ---")

# --- Execution ---
if __name__ == '__main__':
    os.makedirs(INPUT_DIR, exist_ok=True)
    print(f"Ensure your images are in the **'{INPUT_DIR}'** folder.")
    
    # U N C O M M E N T   T O   R U N
    sort_photos()
    # print("\nUNCOMMENT `sort_photos()` to execute the main sorting logic.")
