import sys
import numpy as np
import math
from PySide6.QtWidgets import QApplication

class Line:
    def __init__(self):
        """
        Initialize the Line class with the given tolerances.
        
        Parameters:
        - angle_tolerance_deg: max allowed tilt (default = 1°)
        - max_y_deviation: max allowed y-range (default = 20px)
        """

    def bin_pixels(self, pixels, bin_size=20):
        """
        Groups pixels into 'virtual pixels' by averaging their positions in blocks.
        Args:
        - pixels: List of tuples, [(x1, y1), (x2, y2), ...]
        - bin_size: Number of pixels to group together into one virtual pixel (default 5)
    
        Returns:
        - A list of averaged pixel positions.
        """
        binned_pixels = []
        for i in range(0, len(pixels), bin_size):
            bin_group = pixels[i:i+bin_size]
            avg_x = sum(p[0] for p in bin_group) / len(bin_group)
            avg_y = sum(p[1] for p in bin_group) / len(bin_group)
            binned_pixels.append((avg_x, avg_y))
        return binned_pixels

    # Function to calculate the angle between two vectors (v1, v2) in degrees
    def angle_between_vectors(self, v1, v2):
        # Dot product of v1 and v2
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
        # Magnitudes of v1 and v2
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
        # Cosine of the angle
        cos_theta = dot_product / (mag_v1 * mag_v2)
    
        # Clamp the cosine value to avoid floating point precision issues
        cos_theta = max(-1, min(1, cos_theta))
    
        # Angle in radians, then convert to degrees
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
    
        return angle_deg

    def is_straight(self, A, B, C, D, bin_size=5):

        points = [A, B, C, D]
        binned_points = self.bin_pixels(points, bin_size)

        """
        VERY STRICT CHECK:
        - angle_tolerance_deg: max allowed tilt (default = 1°)
        - max_y_deviation: max allowed y-range (default = 2px)
        """

        # Vectors A-B and B-C (for the angle at point B)
        vector_AB = (B[0] - A[0], B[1] - A[1])
        vector_BC = (C[0] - B[0], C[1] - B[1])

        # Vectors B-C and C-D (for the angle at point C)
        vector_BC2 = (C[0] - B[0], C[1] - B[1])
        vector_CD = (D[0] - C[0], D[1] - C[1])

        # Calculate the angles at B and C
        angle_B = self.angle_between_vectors(vector_AB, vector_BC)
        angle_C = self.angle_between_vectors(vector_BC2, vector_CD)

        # Check if both angles are small enough (under 1 degree)
        if angle_B < 1 and angle_C < 1:
            print("The points form a straight line.")
        else:
            print(f"Angle at point B: {angle_B:.2f} degrees")
            print(f"Angle at point C: {angle_C:.2f} degrees")
            print("The points do not form a straight line.")

        average_degree = (angle_B + angle_C)/2
        y_deviation = math.sqrt((angle_B - ((angle_B + angle_C)/2))**2 + (angle_C - ((angle_B + angle_C)/2))**2)

        # inclination
        if ( average_degree >= 0) and ( average_degree < 5 ):
            inclination = "straight"
        else:
            if ( ( A[1] - B[1] ) < 0 ) and ( ( D[1] - C[1] ) < 0 ) :
                print( A[1] - B[1] )
                print( D[1] - C[1] )
                inclination = "slant up"
            elif ( ( A[1] - B[1] ) > 0 ) and ( ( D[1] - C[1] ) > 0 ) :
                print( A[1] - B[1] )
                print( D[1] - C[1] )
                inclination = "slant down"
            else:
                print( A[1] - B[1] )
                print( D[1] - C[1] )
                inclination = "slant half"
        return {
            "angle 1": angle_B,
            "angle 2": angle_C,
            "avg_deg": average_degree,
            "inclination": inclination,
            "y_deviation": y_deviation
        }

# === Main ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    line = Line()
    sys.exit(app.exec())


