import cv2
import numpy as np
import math

# === CONFIGURATION ===
image_path = 'input_images/images.jpeg'  # Image with shoulder contour line
output_path = 'output_images/images.jpeg'

# === MANUAL INPUT: Define two lines ===
# Shoulder contour line (red): two points
shoulder_p1 = (80, 320)
shoulder_p2 = (420, 270)

# Neck contour line (blue): two points (upright)
neck_p1 = (250, 100)
neck_p2 = (250, 400)

# === DRAW LINES ===
image = cv2.imread(image_path)
cv2.line(image, shoulder_p1, shoulder_p2, (0, 0, 255), 2)  # Red line
cv2.line(image, neck_p1, neck_p2, (255, 0, 0), 2)          # Blue line

# === CALCULATE ANGLE BETWEEN SHOULDER AND NECK LINE ===
def calculate_angle(p1, p2, reference='vertical'):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = abs(math.degrees(angle_rad))
    if reference == 'vertical':
        acute = abs(90 - angle_deg)
    else:
        acute = min(angle_deg, 180 - angle_deg)
    return acute

acute_angle = calculate_angle(shoulder_p1, shoulder_p2, reference='vertical')
shoulder_tilt = 90 - acute_angle

# === ANNOTATE IMAGE ===
cv2.putText(image, f"Tilt: {shoulder_tilt:.2f} deg", (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# === SAVE RESULT ===
cv2.imwrite(output_path, image)
print(f"✅ Shoulder tilt: {shoulder_tilt:.2f} degrees. Saved to {output_path}")

