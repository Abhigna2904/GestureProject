import cv2
import mediapipe as mp
import csv

# ===== MediaPipe setup =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ===== CSV setup =====
csv_file = open("hand_landmarks.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

# header
header = ["frame", "hand", "landmark", "x", "y", "z"]
csv_writer.writerow(header)

# ===== Camera =====
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # mirror view
    frame = cv2.flip(frame, 1)

    # ⭐ IMPORTANT: Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process hand
    result = hands.process(rgb)

    # ===== If hand detected =====
    if result.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):

            # draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # save + print landmarks
            for lm_index, lm in enumerate(hand_landmarks.landmark):
                x, y, z = lm.x, lm.y, lm.z

                # 🔹 print to terminal
                print(f"Frame:{frame_count} Hand:{hand_index} LM:{lm_index} -> {x:.4f}, {y:.4f}, {z:.4f}")

                # 🔹 save to CSV
                csv_writer.writerow([frame_count, hand_index, lm_index, x, y, z])

    # ===== Show camera =====
    cv2.imshow("Hand Detection", frame)

    # press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== Cleanup =====
cap.release()
cv2.destroyAllWindows()
csv_file.close()