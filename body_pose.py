import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

capture = cv2.VideoCapture(1)

CONNECTIONS = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), 
]

while capture.isOpened():
    success, frame = capture.read()
    if success:
        results = model(frame)

        annotatedFrame = frame.copy()

        if len(results[0].keypoints) == 0 or results[0].keypoints is None:
            continue

        keypoints = results[0].keypoints

        if keypoints.conf is None or keypoints.xy is None:
            continue

        confs = keypoints.conf[0].tolist()  
        xys = keypoints.xy[0].tolist() 

        for (start, end) in CONNECTIONS:
            start_conf = confs[start]
            end_conf = confs[end]

            if start_conf < 0.5 or end_conf < 0.5:
                continue

            start_x = int(xys[start][0])
            start_y = int(xys[start][1])
            end_x = int(xys[end][0])
            end_y = int(xys[end][1])
            annotatedFrame = cv2.line(
                annotatedFrame,
                (start_x, start_y),
                (end_x, end_y),
                (255, 0, 255),
                2,
                cv2.LINE_AA
            ) 

        cv2.imshow("YOLOv8 human pose estimation", annotatedFrame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()