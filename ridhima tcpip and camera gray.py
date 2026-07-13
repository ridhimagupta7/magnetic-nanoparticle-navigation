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

def detect_largest_red_blob(frame):
    """Detect red blobs and return centroid of the largest one + mask."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 50:
        return None, mask

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask

def detect_particle(frame):
    """
    Detect a large dark particle cluster.
    More robust against blur and small lighting changes.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Smooth image
    gray = cv2.GaussianBlur(gray, (11, 11), 3)

    # Detect dark objects
    _, mask = cv2.threshold(
        gray,
        120,                    # Try 110-130 if needed
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((7, 7), np.uint8)

    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill holes inside the blob
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Grow blob slightly
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    h, w = mask.shape
    image_center = np.array([w/2, h/2])

    best_contour = None
    best_score = -1

    for c in contours:

        area = cv2.contourArea(c)

        # Ignore tiny blobs
        if area < 300:
            continue

        x, y, cw, ch = cv2.boundingRect(c)

        # Ignore blobs touching image border
        if (
            x <= 5 or
            y <= 5 or
            x + cw >= w - 5 or
            y + ch >= h - 5
        ):
            continue

        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        centroid = np.array([cx, cy])

        # Prefer large blobs near the center
        distance = np.linalg.norm(centroid - image_center)

        score = area - 0.3 * distance

        if score > best_score:
            best_score = score
            best_contour = c

    if best_contour is None:
        return None, mask

    M = cv2.moments(best_contour)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy), mask

# def detect_particle(frame):
#     """
#     Detect a dark/gray particle on a bright background.
#     Works even when the particle becomes blurry.
#     """

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     gray = cv2.GaussianBlur(gray, (9, 9), 0)

#     mask = cv2.adaptiveThreshold(
#         gray,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         31,
#         8
#     )

#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(
#         mask,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     if not contours:
#         return None, mask

#     valid = []

#     for c in contours:

#         area = cv2.contourArea(c)

#         # Ignore tiny noise and huge shadows
#         if 2 < area < 500:
#             valid.append(c)

#     if not valid:
#         return None, mask

#     best_contour = None
#     best_circularity = -1

#     for c in valid:

#         area = cv2.contourArea(c)
#         perimeter = cv2.arcLength(c, True)

#         if perimeter == 0:
#             continue

#         circularity = (
#             4 * np.pi * area /
#             (perimeter * perimeter)
#         )

#         if circularity > best_circularity:
#             best_circularity = circularity
#             best_contour = c

#     if best_contour is None:
#         return None, mask

#     M = cv2.moments(best_contour)

#     if M["m00"] == 0:
#         return None, mask

#     cx = int(M["m10"] / M["m00"])
#     cy = int(M["m01"] / M["m00"])

#     return (cx, cy), mask

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