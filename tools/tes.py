import numpy as np
import matplotlib.pyplot as plt

# New coordinates
x = np.array([59, 118, 175, 233, 257, 293, 350, 408, 465])
y = np.array([219, 231, 239, 251, 257, 263, 274, 282, 294])

# Step 1: Fit a quadratic curve to the data
coefficients = np.polyfit(x, y, 2)  # Fit a second-degree polynomial (quadratic)
a, b, c = coefficients  # Extract the coefficients

# Step 2: Create the fitted quadratic curve
fitted_y = np.polyval(coefficients, x)

# Step 3: Plot the original data and the quadratic fit
plt.plot(x, y, 'o', label="Original Data")
plt.plot(x, fitted_y, '-', label=f"Fitted Quadratic: $f(x) = {a:.2f}x^2 + {b:.2f}x + {c:.2f}$")
plt.title("Quadratic Fit to Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Analyze the sign of 'a' to determine concavity
if a > 0:
    print("The curve is V-shaped (concave up).")
elif a < 0:
    print("The curve is n-shaped (concave down).")
else:
    print("The curve is linear (no concavity).")

