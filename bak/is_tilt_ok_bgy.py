import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
import glob
import math

# --- Configuration ---
INPUT_DIR = 'input_photos'       # Directory where your 100 photos are located
OUTPUT_DIR_PLUS = '42_plus'      # Folder for tilt >= 42 degrees
OUTPUT_DIR_MINUS = '41_minus'    # Folder for tilt < 42 degrees
TILT_THRESHOLD = 42.0            # The angle threshold in degrees

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,       # Optimized for static images
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------------------------------------------------
# --- KEY FUNCTION: 2D 3-Point Angle Calculation for Anchor-Based Tilt ---
# ----------------------------------------------------------------------
def calculate_tilt_angle_3pts(a_coords, b_coords, c_coords):
    """
    Calculates the 2D angle (in degrees) between the line segments BA and BC.
    B: Anchor Point (e.g., Ear)
    A: Horizontal Reference Point (B + 1 unit on X-axis)
    C: Target Point (e.g., Shoulder)
    """
    # Convert to NumPy arrays, using only 2D coordinates (x, y)
    a = np.array(a_coords) 
    b = np.array(b_coords) 
    c = np.array(c_coords) 

    # Calculate vectors
    ba = a - b 
    bc = c - b

    # Dot product and magnitudes for Law of Cosines
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    
    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    try:
        cos_theta = dot_product / (mag_ba * mag_bc)
        # Clamp value to [-1, 1] to prevent math domain error
        cos_theta = np.clip(cos_theta, -1.0, 1.0) 
        
        radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(radians)
        
        return angle_degrees
        
    except Exception:
        return 0.0

# ----------------------------------------------------------------------
# --- Main Processing Function (Uses Anchor-Based Tilt) ---
# ----------------------------------------------------------------------
def process_photo(image_path):
    """
    Loads an image, detects pose landmarks, calculates the Anchor-Based 2D tilt
    of the Left Shoulder relative to the Left Ear.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        screen_landmarks = results.pose_landmarks.landmark
        
        # 1. Define Landmarks for LEFT side (change to RIGHT_EAR/RIGHT_SHOULDER for right side)
        anchor_point = screen_landmarks[mp_pose.PoseLandmark.LEFT_EAR]     # B: Anchor
        target_point = screen_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] # C: Target
        
        # 2. Check visibility
        if anchor_point.visibility < 0.5 or target_point.visibility < 0.5:
            return None
            
        # 3. Extract 2D normalized coordinates (x, y)
        b_coords = np.array([anchor_point.x, anchor_point.y])
        c_coords = np.array([target_point.x, target_point.y])

        # 4. Create Horizontal Reference Point (A)
        # A is one unit to the right of B, maintaining the same Y-coordinate (horizontal line)
        a_coords = np.array([anchor_point.x + 0.1, anchor_point.y]) 
        
        # 5. Calculate the tilt angle
        tilt_angle = calculate_tilt_angle_3pts(a_coords, b_coords, c_coords)
        
        # --- Visualization (Optional: for verification) ---
        img_h, img_w, _ = image.shape
        p_anchor = (int(b_coords[0] * img_w), int(b_coords[1] * img_h))
        p_target = (int(c_coords[0] * img_w), int(c_coords[1] * img_h))
        p_reference = (int(a_coords[0] * img_w), int(a_coords[1] * img_h))
        
        # Draw lines and text for verification
        cv2.line(image, p_anchor, p_target, (0, 255, 0), 2)
        cv2.line(image, p_anchor, p_reference, (255, 0, 0), 2)
        text = f"Tilt: {tilt_angle:.2f} deg"
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(f"annotated_{os.path.basename(image_path)}", image)
        
        return tilt_angle
    else:
        return None

# ----------------------------------------------------------------------
# --- Main Sorting Function ---
# ----------------------------------------------------------------------
def sort_photos():
    """
    Main function to set up folders, process all images, and move them.
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
        
        tilt = process_photo(image_path)
        
        if tilt is not None:
            if tilt >= TILT_THRESHOLD:
                destination_folder = OUTPUT_DIR_PLUS
            else:
                destination_folder = OUTPUT_DIR_MINUS
            
            print(f" -> Tilt {tilt:.2f}. Moving to **{destination_folder}**.")
            
            try:
                # Move the file from INPUT_DIR to the destination folder
                shutil.move(image_path, os.path.join(destination_folder, file_name))
            except Exception as e:
                print(f"Error moving file {file_name}: {e}")
        else:
            print(f" -> Could not determine tilt for {file_name}. Skipping move.")

    print("\n--- Processing Complete ---")

# --- Execution ---
if __name__ == '__main__':
    os.makedirs(INPUT_DIR, exist_ok=True)
    print(f"Ensure your images are in the **'{INPUT_DIR}'** folder.")
    
    # V E R Y   I M P O R T A N T : U N C O M M E N T   T O   R U N
    sort_photos()
    # print("\nUNCOMMENT `sort_photos()` to execute the main sorting logic.")
