import math, cv2, os
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Fix the path to the image by using raw string
src = r"C:\Users\nagab\OneDrive\Documents\image_paranoid\images\output_images\cropper\1762591407217.jpg"

# Debug: Print the absolute path to ensure the path is correct
print("Image Path:", src)

# Check if the image file exists
if not os.path.exists(src):
    print(f"Error: The image file at {src} doesn't exist.")
else:
    # Attempt to load the image
    image = cv2.imread(src)

    if image is None:
        print(f"Error: Failed to load the image from {src}.")
    else:
        print(f"Image loaded successfully from {src}.")

        # Convert the image to RGB (MediaPipe uses RGB images)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize MediaPipe FaceMesh
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            # Process the image and get the landmarks
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Key points for the nose 
                    # (typically nose tip, left nostril, right nostril)

                    # Index 1 is the tip of the nose
                    nose_tip = face_landmarks.landmark[1]

                    # Index 2 is the left nostril
                    nose_left = face_landmarks.landmark[2]

                    # Index 4 is the right nostril
                    nose_right = face_landmarks.landmark[4] 

                    # Convert normalized coordinates to pixel coordinates
                    image_height, image_width, _ = image.shape
                    nose_tip_x = int(nose_tip.x * image_width)
                    nose_tip_y = int(nose_tip.y * image_height)
                    nose_left_x = int(nose_left.x * image_width)
                    nose_left_y = int(nose_left.y * image_height)
                    nose_right_x = int(nose_right.x * image_width)
                    nose_right_y = int(nose_right.y * image_height)

                    # Step 1: Calculate the direction of the nose
                    # (angle between nose_left and nose_right)
                    dx = nose_right_x - nose_left_x
                    dy = nose_right_y - nose_left_y

                    # Calculate the angle of the nose in radians
                    nose_angle = math.atan2(dy, dx)

                    # Step 2: Calculate a line parallel to the nose
                    # Let's use nose_tip as the base and draw a line parallel to the nose.
                    offset_distance = 100  # Distance to offset the line (can adjust as needed)

                    # Line points parallel to the nose (one above, one below)
                    line_p1 = (nose_tip_x + offset_distance * math.cos(nose_angle), 
                               nose_tip_y + offset_distance * math.sin(nose_angle))
                    line_p2 = (nose_tip_x - offset_distance * math.cos(nose_angle), 
                               nose_tip_y - offset_distance * math.sin(nose_angle))

                    # Now you can use these points to draw a line
                    cv2.line(image, (int(line_p1[0]), int(line_p1[1])), 
                             (int(line_p2[0]), int(line_p2[1])), (0, 255, 0), 2)

        # Display the image with the parallel line
        cv2.imshow('Parallel to Nose', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
