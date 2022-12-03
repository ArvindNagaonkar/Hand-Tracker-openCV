import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands= 2, model_complexity=1,detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity        

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img    

    def findPosition(self, img, keyPoint=None, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
            if keyPoint != None:
                print(lmList[keyPoint])
                cv2.circle(img, (cx, cy), 12, (255, 0, 230), cv2.FILLED)
            if keyPoint == None:
                print(lmList)

    # def ParticularKey(self, img, keyPoint=0):
    #     while len(self.lmList) != 0:
    #         print(self.lmList[keyPoint])
    #         cv2.circle(img, (self.cx, self.cy), 12, (255, 0, 230), cv2.FILLED)

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        detector.findPosition(img)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

        cv2.imshow("image", img)
        cv2.waitKey(1)

        
if __name__ == "__main__":
    main()


