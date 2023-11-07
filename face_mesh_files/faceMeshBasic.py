import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video/video2.mp4")
cTime = 0
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

drawSpec = mpDraw.DrawingSpec(thickness = 2, circle_radius = 2)


while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

            for id, lm in enumerate(faceLms.landmark):
                h, w, c = img.shape
                x, y = int(lm.x*w), int(lm.y*h)
                


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'Fps: {int(fps)}', (25,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 3)

    img = cv2.resize(img, (800, 600))
    cv2.imshow('image', img)
    cv2.waitKey(1)