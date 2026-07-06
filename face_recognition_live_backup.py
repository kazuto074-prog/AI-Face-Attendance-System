import face_recognition
import cv2
import pickle
import numpy as np
import time
import csv
import datetime
import os

# ─────────────────────────────────────────
# STEP 1 — Load Face Database
# ─────────────────────────────────────────
print("Loading face database...")

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print(f"Loaded {len(known_names)} face(s): {list(set(known_names))}")
os.makedirs("unknown_faces", exist_ok=True)

# ─────────────────────────────────────────
# STEP 2 — Open Webcam
# ─────────────────────────────────────────
print("Starting webcam... Press Q to quit.")

video = cv2.VideoCapture(0)

# Higher camera quality
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not video.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
TOLERANCE = 0.55
SCALE = 0.20
PROCESS_EVERY_N_FRAMES = 4
UNKNOWN_LABEL = "Unknown"

# Smooth box movement
SMOOTHING = 0.7

# Colors (BGR)
COLOUR_KNOWN = (0, 200, 0)
COLOUR_UNKNOWN = (0, 0, 220)

# ─────────────────────────────────────────
#  ATTENDANCE SETUP
# ─────────────────────────────────────────
ATTENDANCE_FILE = "attendance.csv"
logged_today    = set()   # tracks who was already logged this session

last_unknown_save = 0
UNKNOWN_SAVE_INTERVAL = 30  # seconds
unknown_counter = 0
UNKNOWN_THRESHOLD = 10

# Create CSV with headers if it doesn't exist yet
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Confidence"])

def log_attendance(name, confidence):
    """Log a recognised person once per session."""
    if name in logged_today:
        return   # already logged this person today

    now  = datetime.datetime.now()
    date = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, time_str, f"{confidence:.0f}%"])

    logged_today.add(name)
    print(f"  ✓ Attendance logged: {name} at {time_str}")

# ─────────────────────────────────────────
# INTERNAL VARIABLES
# ─────────────────────────────────────────
frame_count = 0
face_locations = []
face_encodings = []
face_labels = []
smooth_boxes = []

prev_time = time.time()

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
while True:

    success, frame = video.read()

    frame = cv2.flip(frame, 1)

    if not success:
        print("Failed to read from webcam.")
        break

    frame_count += 1

    # Face recognition every N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:

        # Resize frame for faster processing
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=SCALE,
            fy=SCALE
        )

        # Convert BGR → RGB
        rgb_small = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        # Fast HOG detector
        face_locations = face_recognition.face_locations(
            rgb_small,
            model="hog"
        )

        face_encodings = face_recognition.face_encodings(
            rgb_small,
            face_locations
        )

        face_labels = []

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
            known_encodings,
            face_encoding,
            tolerance=TOLERANCE
            )

            distances = face_recognition.face_distance(
            known_encodings,
            face_encoding
            )

            name = UNKNOWN_LABEL
            colour = COLOUR_UNKNOWN
            confidence = 0

            if len(distances) > 0:

                best_match_index = np.argmin(distances)

                if matches[best_match_index]:

                    confidence = (1 - distances[best_match_index]) * 100

                    name = (
                        f"{known_names[best_match_index].upper()} "
                        f"{confidence:.0f}%"
                    )

                    colour = COLOUR_KNOWN

                    log_attendance(
                        known_names[best_match_index].upper(),
                        confidence
                    )

            if name == UNKNOWN_LABEL:

                unknown_counter += 1

                if unknown_counter >= UNKNOWN_THRESHOLD:

                    current_unknown_time = time.time()

                    if current_unknown_time - last_unknown_save > UNKNOWN_SAVE_INTERVAL:

                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                        filename = f"unknown_faces/unknown_{timestamp}.jpg"

                        cv2.imwrite(filename, frame)

                        print(f"Unknown face saved -> {filename}")

                        last_unknown_save = current_unknown_time

            else:
                unknown_counter = 0

            # INSIDE the loop
            face_labels.append((name, colour))

    # ─────────────────────────────────────
    # DRAW SMOOTH BOXES
    # ─────────────────────────────────────
    if len(smooth_boxes) != len(face_locations):
        smooth_boxes = [None] * len(face_locations)

    for i, ((top, right, bottom, left), (name, colour)) in enumerate(
            zip(face_locations, face_labels)):

        # Scale coordinates back
        top = int(top / SCALE)
        right = int(right / SCALE)
        bottom = int(bottom / SCALE)
        left = int(left / SCALE)

        # Smooth movement
        if smooth_boxes[i] is None:
            smooth_boxes[i] = [top, right, bottom, left]
        else:
            old = smooth_boxes[i]

            old[0] = int(old[0] * SMOOTHING + top * (1 - SMOOTHING))
            old[1] = int(old[1] * SMOOTHING + right * (1 - SMOOTHING))
            old[2] = int(old[2] * SMOOTHING + bottom * (1 - SMOOTHING))
            old[3] = int(old[3] * SMOOTHING + left * (1 - SMOOTHING))

        top, right, bottom, left = smooth_boxes[i]

        # Face rectangle
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            colour,
            2
        )

        # Label background
        cv2.rectangle(
            frame,
            (left, bottom + 2),
            (right, bottom + 32),
            colour,
            cv2.FILLED
        )

        # Name text
        cv2.putText(
            frame,
            name,
            (left + 6, bottom + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # ─────────────────────────────────────
    # FPS COUNTER
    # ─────────────────────────────────────
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # HUD
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Faces: {len(face_locations)}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1
    )

    # ─────────────────────────────────────
    # DISPLAY FRAME
    # ─────────────────────────────────────
    cv2.imshow(
        "Face Recognition - Smooth Version",
        frame
    )

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quitting...")
        break

# ─────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────
video.release()
cv2.destroyAllWindows()

print("Camera released. Goodbye!")