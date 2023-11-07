import cv2
import mediapipe as mp
import time

class faceDetector():
    def __init__(self, minDetectionCon = 0.5):

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box #contains the coordinates for the landmarks
                h, w, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h),int(bboxC.width * w),int(bboxC.height * h)
                bboxs.append([id, bbox, detection.score])

                if draw:
                    self.fancyDrawing(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0),4)

        return img, bboxs
    
    def fancyDrawing(self, img, bbox):
        x, y, w, h = bbox
        cv2.rectangle(img, bbox, (255,0,255), 1)
        x1, y1 = x+w, y+h

        l = (int( 0.15 * (w+h)/2   ))

        cv2.line(img, (x,y), (x + l, y), (255,0,255), 20)
        cv2.line(img, (x,y), (x, y + l), (255,0,255), 20)

        cv2.line(img, (x1,y1), (x1 - l, y1), (255,0,255), 20)
        cv2.line(img, (x1,y1), (x1, y1 - l), (255,0,255), 20)

        cv2.line(img, (x,y1), (x + l, y1), (255,0,255), 20)
        cv2.line(img, (x,y1), (x, y1 - l), (255,0,255), 20)

        cv2.line(img, (x1,y), (x1 - l, y), (255,0,255), 20)
        cv2.line(img, (x1,y), (x1, y + l), (255,0,255), 20)



def main():
    cap = cv2.VideoCapture('videos/video2.mp4')
    cTime = 0
    pTime = 0
    bboxs = []
    detector = faceDetector()


    while True:
        success, img = cap.read()
        if not success:
            break

        img, bboxs = detector.findFaces(img)
        print(bboxs)
        
        cTime = time.time()
        fps = 1/ (cTime - pTime)
        pTime = cTime

        img = cv2.resize(img, (800, 600))
        cv2.putText(img, f'FPS: {str(int(fps))}', (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

        cv2.imshow('image', img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()