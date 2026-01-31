import cv2

cap = cv2.VideoCapture("src/data/input_5.mp4")
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
fps_target = 30
out = cv2.VideoWriter("src/data/input_5_30.avi", fourcc, fps_target, (1920, 1080))

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # 60 -> 30 : take every alternate frame
    if frame_index % 2 == 0:
        out.write(frame)

    frame_index += 1

cap.release()
out.release()
print("Done!")
