import cv2
import mediapipe as mp
import time


class faceMeshDetector ():

    def __init__(self, mode = False, numFaces = 2, minDetectCon = 0.5, minTrackCon = 0.5):

        self.mode = mode
        self.numFaces = numFaces
        self.minDetectorCon = minDetectCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh()

        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 2, circle_radius = 2)


    def findFaceMesh(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    x, y = int(lm.x*w), int(lm.y*h)
                    face.append([x,y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture("video/video2.mp4")
    cTime = 0
    pTime = 0

    detector = faceMeshDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img, faces = detector.findFaceMesh(img)

        if len(faces) != 0:
            print(faces[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'Fps: {int(fps)}', (25,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 3)

        img = cv2.resize(img, (800, 600))
        cv2.imshow('image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()