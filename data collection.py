import cv2
import mediapipe as mp
import time
import csv
import os

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Output CSV file
os.makedirs("dataset", exist_ok=True)
csv_file = open("dataset/tap_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["duration", "velocity", "label"])  # Features + Label

print("🟢 Press 'd' to record DOT, 'h' for DASH, 'i' for IDLE, 'q' to quit")

tap_start_time = 0
finger_down = False
prev_y = None
VELOCITY_THRESHOLD = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    fingertip_y = None
    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            fingertip = lm.landmark[8]
            h, w, _ = frame.shape
            fingertip_y = int(fingertip.y * h)
            cv2.circle(frame, (int(fingertip.x * w), fingertip_y), 10, (0, 255, 0), -1)

            if prev_y is not None:
                velocity = fingertip_y - prev_y

                # Downward motion = tap start
                if velocity > VELOCITY_THRESHOLD and not finger_down:
                    tap_start_time = time.time()
                    finger_down = True

                # Upward motion = tap end
                elif velocity < -VELOCITY_THRESHOLD and finger_down:
                    duration = time.time() - tap_start_time
                    finger_down = False
                    print(f"Detected Tap | Duration: {duration:.2f}, Velocity: {velocity:.2f}")

            prev_y = fingertip_y

    cv2.imshow("Collecting Tap Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('d'), ord('h'), ord('i')]:
        label = 'dot' if key == ord('d') else 'dash' if key == ord('h') else 'idle'
        csv_writer.writerow([round(time.time() - tap_start_time, 3), round(velocity, 3), label])
        print(f"✅ Saved: {label.upper()}")

cap.release()
cv2.destroyAllWindows()
csv_file.close()
