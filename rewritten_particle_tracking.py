import socket
import time
import cv2
import numpy as np

LABVIEW_IP = "169.254.66.237"
PORT = 5000

H = np.load("homography.npy")

scale = 0.6

def pixel_to_surface(pt_img, H):
    u, v = float(pt_img[0]), float(pt_img[1])
    pt_h = np.array([u, v, 1.0], dtype=float)
    res = H @ pt_h
    if abs(res[2]) < 1e-9:
        return None
    return (res[0]/res[2], res[1]/res[2])

def detect_particle(frame):
    """
    Detect the center of the largest particle cloud.
    Tunable parameters are grouped together below.
    """

    # ---------- Tunable parameters ----------
    THRESHOLD = 145          # lower -> detect more gray
    MIN_AREA = 500
    BLUR = 9                 # odd number
    DILATE_KERNEL = 25        # merge nearby particles
    DILATE_ITER = 3
    # ---------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR, BLUR), 0)

    _, mask = cv2.threshold(
        gray,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (DILATE_KERNEL, DILATE_KERNEL)
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask = cv2.dilate(mask, kernel, iterations=DILATE_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < MIN_AREA:
        return None, mask

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest], -1, 255, -1)

    # Darkness-weighted centroid
    weights = (255 - gray).astype(np.float32)
    weights *= contour_mask / 255.0

    total = np.sum(weights)
    if total == 0:
        return None, mask

    Y, X = np.indices(gray.shape)
    cx = int(np.sum(X * weights) / total)
    cy = int(np.sum(Y * weights) / total)

    return (cx, cy), mask

def connect_to_labview(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.connect((ip, port))
    return s

sock = connect_to_labview(LABVIEW_IP, PORT)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    centroid, mask = detect_particle(frame)

    if centroid is not None:
        cv2.circle(frame, centroid, 6, (0,0,255), -1)

        XY = pixel_to_surface(centroid, H)
        if XY is not None:
            X, Y = XY

            cv2.putText(
                frame,
                f"Pixel {centroid} -> Surface ({X:.2f}, {Y:.2f})",
                (centroid[0]+10, centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0,255,0),
                2
            )

            msg = f"{X:+08.3f},{Y:+08.3f}\n"
            sock.sendall(msg.encode("utf-8"))
            print("Centroid:", centroid,
                  " Sent:", msg.strip())

    cv2.imshow("Webcam",
               cv2.resize(frame,None,fx=0.6,fy=0.6))
    cv2.imshow("Particle Mask",
               cv2.resize(mask,None,fx=0.6,fy=0.6))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
sock.close()
cv2.destroyAllWindows()
