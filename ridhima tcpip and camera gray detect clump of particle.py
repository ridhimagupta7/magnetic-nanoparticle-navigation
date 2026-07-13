import socket
import time
import cv2
import numpy as np

# ===============================
# TCP/IP Settings (LabVIEW server)
# ===============================
LABVIEW_IP = "169.254.66.237" #"192.168.73.186"
PORT = 5000

# ===============================
# Homography (pixel -> surface)
# ===============================

H = np.load("homography.npy")
# H = np.array([
#    [ 4.70398723e-02,  1.06894660e-03, -2.95010491e+01],
#    [ -1.28438650e-03,  4.70957545e-02, -2.50452698e+01],
#    [5.49132930e-05,  5.49948949e-06,  1.00000000e+00]
#], dtype=float)

# HSV thresholds for red
lower_red1 = np.array([0, 60, 50])
upper_red1 = np.array([12, 255, 255])
lower_red2 = np.array([160, 60, 50])
upper_red2 = np.array([180, 255, 255])

kernel = np.ones((5, 5), np.uint8)

# Display scaling
scale = 0.6

def pixel_to_surface(pt_img, H):
    """Convert pixel (u,v) → surface (X,Y) using homography."""
    u, v = float(pt_img[0]), float(pt_img[1])
    pt_h = np.array([u, v, 1.0], dtype=float)
    res = H @ pt_h
    if abs(res[2]) < 1e-9:
        return None
    X = res[0] / res[2]
    Y = res[1] / res[2]
    return (X, Y)

def detect_particle(frame):
    """
    Detect the largest blurry particle cloud.
    """

    # ----------------------------
    # Convert to grayscale
    # ----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Smooth image
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    # ----------------------------
    # Adaptive threshold
    # ----------------------------
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,      # neighborhood size (must be odd)
        5        # threshold offset (try 3-10)
    )

    # ----------------------------
    # Morphology
    # ----------------------------
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11,11)
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # ----------------------------
    # Find contours
    # ----------------------------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, mask

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 300:
        return None, mask

    # ----------------------------
    # Weighted centroid
    # ----------------------------
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest], -1, 255, -1)

    weights = (255 - gray).astype(np.float32)
    weights *= contour_mask / 255.0

    total = np.sum(weights)

    if total == 0:
        return None, mask

    Y, X = np.indices(weights.shape)

    cx = int(np.sum(X * weights) / total)
    cy = int(np.sum(Y * weights) / total)

    return (cx, cy), mask

def connect_to_labview(ip: str, port: int) -> socket.socket:
    """Create and connect a TCP socket to LabVIEW."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # lower latency
    sock.connect((ip, port))
    return sock

# ===============================
# Main
# ===============================
sock = None
cap = None

try:
    # --- Connect to LabVIEW ---
    sock = connect_to_labview(LABVIEW_IP, PORT)
    print(f"🔗 Connected to LabVIEW at {LABVIEW_IP}:{PORT}")

    # --- Open webcam ---
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access the webcam.")

    print("Press 'q' to quit.")
    print("Sending format: X,Y\\n\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        centroid, mask = detect_particle(frame)

        msg_to_send = None

        if centroid is not None:
            cv2.circle(frame, centroid, 6, (0, 0, 255), -1)

            XY = pixel_to_surface(centroid, H)
            if XY is not None:
                X, Y = XY

                # Overlay
                cv2.putText(
                    frame,
                    f"Pixel {centroid} -> Surface ({X:.2f}, {Y:.2f})",
                    (centroid[0] + 10, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                # ---- Build TCP message (NO time) ----
                #msg_to_send = f"{X:.4f},{Y:.4f}\n"
                #msg_to_send = "111.111,222.222\n"
                msg_to_send = f"{X:+08.3f},{Y:+08.3f}\n"

        centroid, mask = detect_particle(frame)
        print("Centroid:", centroid)
        # Send TCP + print what is sent
        if msg_to_send is not None:
            try:
                sock.sendall(msg_to_send.encode("utf-8"))
                time.sleep(0.1)
                print(f"Sent to LabVIEW: {msg_to_send.strip()}")
            except Exception as e:
                print(f"❌ TCP send failed: {e}")
                break

        # Resize ONLY for display (60%)
        display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        display_mask = cv2.resize(mask, None, fx=scale, fy=scale)

        cv2.imshow("Webcam", display_frame)
        cv2.imshow("Red Mask", display_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    if sock is not None:
        try:
            sock.close()
        except:
            pass

    print("✅ Closed camera + TCP connection.")