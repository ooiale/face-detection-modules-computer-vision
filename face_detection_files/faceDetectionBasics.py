import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/video.mp4')
cTime = 0
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection( min_detection_confidence=0.1)


while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box #contains the coordinates for the landmarks
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h),int(bboxC.width * w),int(bboxC.height * h)
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0),4)

    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime

    img = cv2.resize(img, (800, 600))
    cv2.putText(img, f'FPS: {str(int(fps))}', (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

    cv2.imshow('image', img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()