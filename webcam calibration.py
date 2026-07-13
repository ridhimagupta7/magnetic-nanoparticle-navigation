import cv2
import numpy as np

def get_centroid_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
    return centroids

# ===============================
# Initialize webcam
# ===============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("❌ Cannot access the webcam.")
    exit()

# ===============================
# Manual focus setup
# ===============================
try:
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 170)
    focus_val = 170
    print("📷 Manual focus control enabled. Starting focus set to 170.")
except:
    focus_val = 170
    print("⚠️ Manual focus control may not be supported on this webcam.")

print("🎥 Streaming started. Press 'q' to quit.")
print("Use '+' or '-' to adjust focus.\n")

# ===============================
# Main Loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color thresholds
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_DILATE, kernel)

    # Detect centroids
    centroids = get_centroid_from_mask(mask_clean)

    for i, (cx, cy) in enumerate(centroids, start=1):
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{i}", (cx + 10, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"[{i}] Centroid at pixel: ({cx}, {cy})")

    cv2.putText(frame, f"Focus: {focus_val:.0f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # ===============================
    # Resize ONLY for display (Option 1)
    # ===============================
    scale = 0.6  # change this if you want smaller/larger window
    display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    display_mask = cv2.resize(mask_clean, None, fx=scale, fy=scale)

    cv2.imshow("Live Stream - Detected Red Circles", display_frame)
    cv2.imshow("Mask", display_mask)

    key = cv2.waitKey(1) & 0xFF

    # Manual Focus Adjustment
    if key == ord('+') or key == ord('='):
        focus_val = min(255, focus_val + 5)
        cap.set(cv2.CAP_PROP_FOCUS, focus_val)
        print(f"🔧 Focus increased to {focus_val}")

    elif key == ord('-') or key == ord('_'):
        focus_val = max(0, focus_val - 5)
        cap.set(cv2.CAP_PROP_FOCUS, focus_val)
        print(f"🔧 Focus decreased to {focus_val}")

    elif key == ord('q'):
        print("🔚 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()