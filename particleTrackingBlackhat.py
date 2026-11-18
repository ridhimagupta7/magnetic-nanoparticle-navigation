import socket
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# TCP/IP settings
# ============================================================

LABVIEW_IP = "169.254.66.237"
PORT = 5000


# ============================================================
# Homography file
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

HOMOGRAPHY_PATH = (
    SCRIPT_DIR
    / "magnetic-nanoparticle-navigation"
    / "homography.npy"
)

if not HOMOGRAPHY_PATH.exists():
    raise FileNotFoundError(
        f"Could not find homography.npy at:\n{HOMOGRAPHY_PATH}\n\n"
        "Run the calibration program first, or move homography.npy "
        "into the magnetic-nanoparticle-navigation folder."
    )

H = np.load(HOMOGRAPHY_PATH)

if H.shape != (3, 3):
    raise ValueError(
        f"Expected a 3x3 homography matrix, but got shape {H.shape}."
    )


# ============================================================
# Single-particle detection settings
# ============================================================

# Black-hat morphology emphasizes a small dark object against
# a slowly changing bright background.
BLACKHAT_KERNEL_SIZE = 31

# Lower this if the particle disappears from the mask.
# Raise it if too many faint marks appear.
BLACKHAT_THRESHOLD = 8

# Expected particle size, in pixels squared.
MIN_PARTICLE_AREA = 5
MAX_PARTICLE_AREA = 800

# Reject contours touching image borders.
BORDER_MARGIN = 10

# Reject very thin scratches while allowing a blurry round particle.
MIN_CIRCULARITY = 0.15

# Display scaling.
DISPLAY_SCALE = 0.6


# ============================================================
# Coordinate conversion
# ============================================================

def pixel_to_surface(pixel_point, homography):
    """Convert pixel coordinates into calibrated surface coordinates."""

    u, v = map(float, pixel_point)
    pixel_h = np.array([u, v, 1.0], dtype=float)
    result = homography @ pixel_h

    if abs(result[2]) < 1e-9:
        return None

    x_surface = result[0] / result[2]
    y_surface = result[1] / result[2]

    return x_surface, y_surface


# ============================================================
# Particle detector
# ============================================================

def detect_particle(frame):
    """
    Detect one small dark gray/black particle on a light background.

    Returns:
        centroid: (x, y), or None when no valid particle is found
        mask: binary mask used for debugging
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Small blur reduces camera noise without smearing the particle too much.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Black-hat highlights locally dark regions.
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BLACKHAT_KERNEL_SIZE, BLACKHAT_KERNEL_SIZE)
    )

    blackhat = cv2.morphologyEx(
        blurred,
        cv2.MORPH_BLACKHAT,
        blackhat_kernel
    )

    _, mask = cv2.threshold(
        blackhat,
        BLACKHAT_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Remove isolated pixel noise and fill small gaps.
    cleanup_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cleanup_kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cleanup_kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    frame_height, frame_width = mask.shape
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if not (MIN_PARTICLE_AREA <= area <= MAX_PARTICLE_AREA):
            continue

        x, y, width, height = cv2.boundingRect(contour)

        # Ignore tray edges and objects cut off by the frame.
        if (
            x <= BORDER_MARGIN
            or y <= BORDER_MARGIN
            or x + width >= frame_width - BORDER_MARGIN
            or y + height >= frame_height - BORDER_MARGIN
        ):
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        if circularity < MIN_CIRCULARITY:
            continue

        valid_contours.append(contour)

    if not valid_contours:
        return None, mask

    # The real particle is the largest remaining valid dark blob.
    particle_contour = max(valid_contours, key=cv2.contourArea)

    moments = cv2.moments(particle_contour)

    if moments["m00"] == 0:
        return None, mask

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return (cx, cy), mask


# ============================================================
# TCP connection
# ============================================================

def connect_to_labview(ip, port):
    """Connect to the LabVIEW TCP server."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(10.0)
    sock.connect((ip, port))
    sock.settimeout(None)

    return sock


# ============================================================
# Main program
# ============================================================

def main():
    sock = None
    cap = None

    try:
        sock = connect_to_labview(LABVIEW_IP, PORT)
        print(f"Connected to LabVIEW at {LABVIEW_IP}:{PORT}")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)

        if not cap.isOpened():
            raise RuntimeError("Cannot access the webcam.")

        # Lock camera settings where supported.
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 170)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)

        print(f"Loaded homography from:\n{HOMOGRAPHY_PATH}")
        print("Press q while an OpenCV window is selected to quit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame.")
                break

            # Detect exactly once per frame.
            centroid, mask = detect_particle(frame)

            if centroid is not None:
                cv2.circle(
                    frame,
                    centroid,
                    6,
                    (0, 0, 255),
                    -1
                )

                surface = pixel_to_surface(centroid, H)

                if surface is not None:
                    x_surface, y_surface = surface

                    cv2.putText(
                        frame,
                        (
                            f"Pixel {centroid} -> Surface "
                            f"({x_surface:.2f}, {y_surface:.2f})"
                        ),
                        (centroid[0] + 10, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    # Exactly 18 bytes:
                    # 8 for X, 1 comma, 8 for Y, 1 newline.
                    message = (
                        f"{x_surface:+08.3f},"
                        f"{y_surface:+08.3f}\n"
                    )

                    if len(message.encode("utf-8")) != 18:
                        raise RuntimeError(
                            f"TCP message was not 18 bytes: {message!r}"
                        )

                    sock.sendall(message.encode("utf-8"))

                    print(
                        f"Centroid: {centroid} | "
                        f"Sent to LabVIEW: {message.strip()}"
                    )
            else:
                cv2.putText(
                    frame,
                    "Particle not detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            display_frame = cv2.resize(
                frame,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE
            )

            display_mask = cv2.resize(
                mask,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE
            )

            cv2.imshow(
                "Webcam - Single Dark Particle",
                display_frame
            )

            cv2.imshow(
                "Particle Mask",
                display_mask
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                break

    except KeyboardInterrupt:
        print("Stopped with Ctrl+C.")

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

        print("Closed camera and TCP connection.")


if __name__ == "__main__":
    main()