import cv2
import numpy as np
import os

# === CONFIGURATION ===
source_folder = "facial_thirds_imgs/Ridwan Kamil/frontal"    # Folder with input images
target_folder = "is_narrow_imgs/Ridwan Kamil/frontal"    # Folder to save annotated images
os.makedirs(target_folder, exist_ok=True)

# --- Detection parameters ---
MIN_CONTOUR_AREA = 200  # ignore tiny blobs
LINE_THICKNESS = 2      # for annotation

# --- HSV color ranges ---
COLOR_RANGES = {
    "red": [((0, 120, 80), (10, 255, 255)), ((170, 120, 80), (180, 255, 255))],  # dual red range
    "yellow": [((20, 120, 120), (40, 255, 255))],
    "blue": [((90, 80, 70), (130, 255, 255))]
}

# --- Helper functions ---
def build_mask(hsv, ranges):
    """Combine multiple HSV ranges for one color."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (low, high) in ranges:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    # small morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def find_line_y(mask):
    """Return the y-coordinate of the main horizontal line in a mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] > 0:
        return int(M["m01"] / M["m00"])
    ys = largest[:, 0, 1]
    return int(np.median(ys)) if len(ys) > 0 else None


# === MAIN LOOP ===
for filename in sorted(os.listdir(source_folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(source_folder, filename)
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Could not read {filename}")
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    ys = {}
    for color, ranges in COLOR_RANGES.items():
        mask = build_mask(hsv, ranges)
        ys[color] = find_line_y(mask)

    print(f"\n🖼️ {filename}")
    for color in ["red", "yellow", "blue"]:
        print(f"  {color:6s} line y = {ys[color]}")

    if None in ys.values():
        print("  ❌ Missing one or more color lines — cannot compute index.")
        out_path = os.path.join(target_folder, filename)
        cv2.imwrite(out_path, img)
        continue

    # Compute distances and index
    d1 = abs(ys["red"] - ys["yellow"])
    d2 = abs(ys["yellow"] - ys["blue"])
    index = d1 / d2 if d2 != 0 else float("inf")

    short_middle_face = index < 1
    print(f"  👉 index = {index:.3f}")
    print(f"  short middle face is {short_middle_face}")

    # Annotate image
    annotated = img.copy()
    cv2.line(annotated, (0, ys["red"]), (w, ys["red"]), (0, 0, 255), LINE_THICKNESS)
    cv2.line(annotated, (0, ys["yellow"]), (w, ys["yellow"]), (0, 255, 255), LINE_THICKNESS)
    cv2.line(annotated, (0, ys["blue"]), (w, ys["blue"]), (255, 0, 0), LINE_THICKNESS)

    cv2.putText(annotated, f"Index: {index:.3f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    txt = f"Short middle face: {short_middle_face}"
    cv2.putText(annotated, txt, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    out_path = os.path.join(target_folder, filename)
    cv2.imwrite(out_path, annotated)

print("\n✅ Processing complete. Results saved in:", os.path.abspath(target_folder))
