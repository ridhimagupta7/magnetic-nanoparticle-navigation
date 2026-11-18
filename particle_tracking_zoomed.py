"""Track one small dark magnetic particle with a STATIONARY, zoomed-in camera.

Loads the global-mm calibration produced by calibration_corner.py (homography.npy).
The homography already maps pixels to the workspace-centered [-20, +20] mm
coordinate system at the initial/calibration camera position.

Controls (click the video window first):
    Left click: select the actual particle near the click
    R:          reset/reselect automatically
    D:          show/hide detection mask
    +/-:        increase/decrease black-hat threshold (+ = stricter)
    L:          connect/disconnect LabVIEW TCP streaming
    H:          reload homography.npy from disk
    Q or ESC:   quit

Requires: pip install opencv-python numpy
Run:      python particle_tracking_zoomed.py

IMPORTANT: This version assumes the camera does NOT move. Camera motion
correction will be added later, after LabVIEW motor positions are available.
"""

from pathlib import Path
import socket
import sys
import time

import cv2
import numpy as np

# ======================== USER SETTINGS ========================
LABVIEW_IP = "169.254.66.237"
PORT = 5000
START_WITH_LABVIEW = False       # False = test camera without LabVIEW; press L to connect
SEND_INTERVAL_SEC = 0.05        # <= 20 position messages per second

H_PATH = Path(__file__).resolve().parent / "homography.npy"
CAMERA_INDEX = 0
WIDTH, HEIGHT = 1280, 720
FPS = 60
DISPLAY_SCALE = 0.6           # only changes displayed window; math uses full pixels

# Small dark particle on a mostly white/light surface.
# These are starting values; adjust while watching detection-mask window.
BLACKHAT_KERNEL = 41           # odd, bigger than particle's apparent diameter
BLACKHAT_THRESHOLD = 12        # '+' increases strictness; '-' detects fainter particles
MIN_PARTICLE_AREA = 5          # mask pixels
MAX_PARTICLE_AREA = 1200       # increase if particle appears very large at new zoom
MIN_CONTRAST = 10.0            # particle darker than local background
MAX_ASPECT_RATIO = 4.0         # avoid long scratches
MAX_TRACK_JUMP_PX = 170        # maximum displacement between adjacent processed frames
MAX_CLICK_DISTANCE_PX = 90     # how close a candidate must be to user click
RED_MASK_PADDING = 8          # excludes printed red reference dots

WORKSPACE_MIN_MM = -20.0
WORKSPACE_MAX_MM = 20.0
# Camera is stationary for this phase. Do not double-apply calibration-grid offset!
camera_x_mm = 0.0
camera_y_mm = 0.0

WINDOW = "Magnetic particle tracking - zoomed stationary camera"
MASK_WINDOW = "Particle detection mask"
# ===============================================================


def load_homography():
    if not H_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot find {H_PATH}\n"
            "Run calibration_corner.py and press SPACE, then put "
            "homography.npy in the SAME FOLDER as this tracker."
        )
    H = np.load(H_PATH)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError("homography.npy must be a finite 3x3 matrix")
    return H


def pixel_to_global_mm(pixel, H):
    u, v = pixel
    result = H @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    if abs(result[2]) < 1e-10:
        return None
    # Camera offsets remain zero while the camera is stationary.
    return (float(result[0] / result[2] + camera_x_mm),
            float(result[1] / result[2] + camera_y_mm))


def open_camera():
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera. Close calibration.py first.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap


def detect_candidates(frame, blackhat_kernel, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (blackhat_kernel, blackhat_kernel))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # Ignore red calibration markers if they remain in the camera view.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_a = cv2.inRange(hsv, (0, 75, 50), (12, 255, 255))
    red_b = cv2.inRange(hsv, (165, 75, 50), (179, 255, 255))
    red = cv2.bitwise_or(red_a, red_b)
    pad_kernel = np.ones((2 * RED_MASK_PADDING + 1,) * 2, np.uint8)
    red = cv2.dilate(red, pad_kernel)
    mask[red > 0] = 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    h, w = gray.shape
    for i in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if not MIN_PARTICLE_AREA <= area <= MAX_PARTICLE_AREA:
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        if max(bw / max(bh, 1), bh / max(bw, 1)) > MAX_ASPECT_RATIO:
            continue

        region = labels[y:y + bh, x:x + bw] == i
        dark_values = gray[y:y + bh, x:x + bw][region]
        if dark_values.size == 0:
            continue
        padding = max(12, max(bw, bh))
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(w, x + bw + padding), min(h, y + bh + padding)
        background = float(np.percentile(gray[y0:y1, x0:x1], 70))
        contrast = background - float(np.mean(dark_values))
        if contrast < MIN_CONTRAST:
            continue

        bh_strength = float(np.mean(blackhat[y:y + bh, x:x + bw][region]))
        cx, cy = [float(v) for v in centroids[i]]
        # Score darkness more than region size, so larger dust specks aren't
        # automatically preferred over a small black particle.
        score = 1.7 * contrast + 0.8 * bh_strength + 0.1 * min(area, 60)
        candidates.append((cx, cy, score, area))

    return candidates, mask


def select_particle(candidates, last_position, manual_click=None):
    if not candidates:
        return None
    if manual_click is not None:
        near = sorted(candidates,
                      key=lambda c: np.hypot(c[0] - manual_click[0],
                                             c[1] - manual_click[1]))
        candidate = near[0]
        if np.hypot(candidate[0] - manual_click[0],
                    candidate[1] - manual_click[1]) <= MAX_CLICK_DISTANCE_PX:
            return candidate
        return None
    if last_position is None:
        return max(candidates, key=lambda c: c[2])
    nearby = [c for c in candidates if np.hypot(c[0] - last_position[0],
                                                c[1] - last_position[1])
              <= MAX_TRACK_JUMP_PX]
    if not nearby:
        return None  # Never silently jump to a distant unrelated dark spot.
    return max(nearby, key=lambda c: c[2] - 2.0 * np.hypot(
        c[0] - last_position[0], c[1] - last_position[1]))


def connect_labview():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(0.5)
    try:
        sock.connect((LABVIEW_IP, PORT))
        sock.settimeout(0.2)
        print(f"LabVIEW connected: {LABVIEW_IP}:{PORT}")
        return sock
    except OSError as exc:
        sock.close()
        print(f"LabVIEW not connected ({exc}). Tracking still works.")
        return None


def main():
    H = load_homography()
    cap = open_camera()
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {frame_w}x{frame_h} | homography: {H_PATH}")
    print("Stationary camera only; global origin = workspace center in mm.")
    print("Click particle to select | R reset | D mask | +/- threshold |")
    print("L connect/disconnect LabVIEW | H reload calibration | Q quit")

    state = {"last": None, "click": None, "show_mask": True}
    threshold = BLACKHAT_THRESHOLD
    sock = connect_labview() if START_WITH_LABVIEW else None
    last_sent = 0.0
    last_logged = 0.0
    latest_coords = None

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x / DISPLAY_SCALE, y / DISPLAY_SCALE)
            state["last"] = None
            print(f"Particle selection requested near {state['click']} (full pixels)")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame.")
                break

            candidates, mask = detect_candidates(frame, BLACKHAT_KERNEL, threshold)
            candidate = select_particle(candidates, state["last"], state["click"])
            if state["click"] is not None:
                if candidate is None:
                    print("No dark particle found near click. Try lower threshold (-), or click again.")
                state["click"] = None

            overlay = frame.copy()
            for x, y, score, area in candidates:
                cv2.circle(overlay, (round(x), round(y)), 5, (150, 150, 150), 1)

            status = "SEARCHING (click particle if auto selection is wrong)"
            status_color = (0, 180, 255)
            latest_coords = None
            if candidate is not None:
                px, py, _, _ = candidate
                state["last"] = (px, py)
                latest_coords = pixel_to_global_mm((px, py), H)
                cv2.circle(overlay, (round(px), round(py)), 16, (0, 255, 0), 2)
                cv2.drawMarker(overlay, (round(px), round(py)), (0, 255, 0),
                               cv2.MARKER_CROSS, 22, 2)

                if latest_coords is not None:
                    X, Y = latest_coords
                    in_workspace = (WORKSPACE_MIN_MM <= X <= WORKSPACE_MAX_MM
                                    and WORKSPACE_MIN_MM <= Y <= WORKSPACE_MAX_MM)
                    status = (f"Particle pixel=({px:.1f},{py:.1f})  "
                              f"global mm=({X:+.2f},{Y:+.2f})")
                    if not in_workspace:
                        status += "  OUTSIDE WORKSPACE (not sending)"
                        status_color = (0, 180, 255)
                    else:
                        status_color = (0, 255, 0)
                        now = time.monotonic()
                        if sock is not None and now - last_sent >= SEND_INTERVAL_SEC:
                            msg = f"{X:+08.3f},{Y:+08.3f}\n"
                            if len(msg.encode("ascii")) == 18:
                                try:
                                    sock.sendall(msg.encode("ascii"))
                                    last_sent = now
                                except OSError as exc:
                                    print(f"LabVIEW send failed: {exc}. Press L to reconnect.")
                                    sock.close()
                                    sock = None
                        if now - last_logged >= 0.3:
                            print(status)
                            last_logged = now
            elif state["last"] is not None:
                status = "LOST: move particle near last location, click it, or press R"
                cv2.circle(overlay, tuple(round(v) for v in state["last"]),
                           18, (0, 180, 255), 1)

            cv2.rectangle(overlay, (0, 0), (frame_w, 75), (18, 18, 18), -1)
            cv2.putText(overlay, status, (12, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        status_color, 2, cv2.LINE_AA)
            tcp_status = "CONNECTED" if sock is not None else "OFF"
            second_line = (f"Threshold={threshold} | candidates={len(candidates)} "
                           f"| LabVIEW={tcp_status} | click=select R=reset "
                           f"+/-=threshold D=mask L=TCP H=reload Q=quit")
            cv2.putText(overlay, second_line, (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (225, 225, 225), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, cv2.resize(overlay, None,
                       fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                       interpolation=cv2.INTER_AREA))
            if state["show_mask"]:
                cv2.imshow(MASK_WINDOW, cv2.resize(mask, None,
                           fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                           interpolation=cv2.INTER_NEAREST))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                state["last"] = None
                state["click"] = None
                print("Reset: looking for particle anywhere in the camera frame")
            elif key == ord("d"):
                state["show_mask"] = not state["show_mask"]
                if not state["show_mask"]:
                    cv2.destroyWindow(MASK_WINDOW)
            elif key in (ord("+"), ord("=")):
                threshold = min(100, threshold + 2)
                print(f"Blackhat threshold: {threshold}")
            elif key in (ord("-"), ord("_")):
                threshold = max(1, threshold - 2)
                print(f"Blackhat threshold: {threshold}")
            elif key == ord("l"):
                if sock is not None:
                    sock.close()
                    sock = None
                    print("LabVIEW disconnected; continuing video tracking")
                else:
                    sock = connect_labview()
            elif key == ord("h"):
                try:
                    H = load_homography()
                    state["last"] = None
                    print("Reloaded homography.npy successfully")
                except (OSError, ValueError) as exc:
                    print(f"Cannot reload homography: {exc}")
    finally:
        cap.release()
        if sock is not None:
            sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
