import cv2
from ultralytics import YOLO

# 모델의 로드. 자세 추론용 모델 데이터를 로드합니다.
model = YOLO("yolov8n-pose.pt")

# 본체의 웹캠에서 캡처하도록 설정합니다.
video_path = 1  # 본체에 내장된 카메라를 지정합니다.
capture = cv2.VideoCapture(video_path)

# 키포인트 위치별 명칭 정의
KEYPOINTS_NAMES = [
    "nose",  # 0
    "eye(L)",  # 1
    "eye(R)",  # 2
    "ear(L)",  # 3
    "ear(R)",  # 4
    "shoulder(L)",  # 5
    "shoulder(R)",  # 6
    "elbow(L)",  # 7
    "elbow(R)",  # 8
    "wrist(L)",  # 9
    "wrist(R)",  # 10
    "hip(L)",  # 11
    "hip(R)",  # 12
    "knee(L)",  # 13
    "knee(R)",  # 14
    "ankle(L)",  # 15
    "ankle(R)",  # 16
]

while capture.isOpened():
    success, frame = capture.read()
    if success:
        # 추론을 실행합니다.
        results = model(frame)

        annotatedFrame = results[0].plot()

        # 검출된 객체의 이름과 바운딩 박스 좌표를 가져옵니다.
        names = results[0].names
        classes = results[0].boxes.cls
        boxes = results[0].boxes

        for box, cls in zip(boxes, classes):
            name = names[int(cls)]
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]

        if len(results[0].keypoints) == 0 or results[0].keypoints is None:
            continue

        # 자세 분석 결과의 키포인트를 가져옵니다.
        keypoints = results[0].keypoints

        # 키포인트가 None인지 확인하고 처리합니다.
        if keypoints.conf is None or keypoints.xy is None:
            continue
        
        confs = keypoints.conf[0].tolist()  # 추론 결과: 1에 가까울수록 신뢰도가 높습니다.
        xys = keypoints.xy[0].tolist()  # 좌표

        for index, keypoint in enumerate(zip(xys, confs)):
            score = keypoint[1]

            # 스코어가 0.5 이하이면 그리지 않습니다.
            if score < 0.5:
                continue

            x = int(keypoint[0][0])
            y = int(keypoint[0][1])
            print(
                f"Keypoint Name={KEYPOINTS_NAMES[index]}, X={x}, Y={y}, Score={score:.4}"
            )

        cv2.imshow("YOLOv8 human pose estimation", annotatedFrame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
