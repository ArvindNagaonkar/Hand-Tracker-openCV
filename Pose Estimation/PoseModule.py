import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, upBody=False, model_complexity=1, smooth= True, detectionCon = 0.5, trackingCon = 0.5 ):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.model_complexity = model_complexity 
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody,self.model_complexity, self.smooth, self.detectionCon, self.trackingCon)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB) 
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, keyPoint = None):
        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
            if keyPoint != None:
                print(lmList[keyPoint])
                cv2.circle(img, (lmList[keyPoint][1],lmList[keyPoint][2]), 5, (255, 0, 0), cv2.FILLED)
            if keyPoint == None:
                    print(lmList)
    



def main():
    cap =cv2.VideoCapture('/home/arvind/Desktop/openCVProject-env/Pose Estimation/videos/6.mp4', )
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img =  detector.findPose(img)
        detector.getPosition(img, 23)
        cTime = time.time()
        fps = 1/ (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (25, 0,0), 3)
        cv2.imshow("video", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()