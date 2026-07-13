import cv2
import numpy as np

# Real-world coordinates (X, Y) in mm — Ground Truth
pts_real = np.array([
    [0, -15],
    [-15, 0],
    [0, 0],
    [15, 0],
    [0, 15]
], dtype=np.float32)

# Pixel coordinates from webcam (u, v)
pts_img = np.array([
    [622,219],   # Top
    [290,539],   # Left
    [615,550],   # Center
    [951,557],   # Right
    [607,879]    # Bottom
], dtype=np.float32)

# Compute homography matrix H
H, status = cv2.findHomography(pts_img, pts_real)
print("Homography matrix:\n", H)

# Check calibration by transforming each pixel point and comparing with ground truth
print("\n--- Calibration Check ---")
for i in range(len(pts_img)):
    pt_img = np.append(pts_img[i], 1)  # Convert to homogeneous
    projected = H @ pt_img             # Matrix multiply
    projected /= projected[2]          # Normalize homogeneous coordinate

    predicted_real = projected[:2]
    true_real = pts_real[i]
    error = predicted_real - true_real

    print(f"Point {i+1}:")
    print("Predicted:", predicted_real)
    print("Ground Truth:", true_real)
    print("Error (mm):", error)
    print()
