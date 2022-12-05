import cv2 
import numpy as np 
import time
import os
import HandtrackingModule as htm


folderPath = "/home/arvind/Desktop/openCVProject-env/Virtual Painter/Header"

myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))
header = overlayList[0]
drawColor = (0, 0, 0)
brushThickness = 12
eraserThickness = 50

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
detector = htm.handDetector(detectionCon=0.85)

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # find the Landmark

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        # tip of index and middle finger
        x1 ,y1 = lmList[8][1], lmList[8][2]
        x2 ,y2 = lmList[12][1], lmList[12][2]

        # Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        # If Selection mode - 2 fingers are up

        if fingers[1] and fingers[2]:
            # print('Selection Mode')
            # Checking for Click
            xp, yp = 0, 0

            if y1 < 125:

                if 267 <= x1 <= 366:
                    header = overlayList[2]
                    drawColor = (0, 0, 255)
                if 434 <= x1 <= 496:
                    header = overlayList[3]
                    drawColor = (196, 102, 255)
                if 600 <= x1 <= 660:
                    header = overlayList[5]
                    drawColor = (235, 23, 94)
                if 757 <= x1 <= 817:
                    header = overlayList[4]   
                    drawColor = (87, 217, 126)
                if 920 <= x1 <= 980:
                    header = overlayList[1]
                    drawColor = (89, 222, 255)
                if 1116  <= x1 <= 1176:
                    header = overlayList[6]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1-25), (x2, y2 +25), drawColor, cv2.FILLED)
            
        # If Drawing Mode - Inder finger is up 
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            # print('Drawing Mode')

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) 
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)   

    # Setting header Img
    img[0:125,0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Video", img)
    cv2.imshow("Image canvas", imgCanvas)
    cv2.waitKey(1)