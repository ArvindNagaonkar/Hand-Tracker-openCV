import cv2
import time
import numpy as np
import HandtrackingModule as htm
import math
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4,hCam)
pTime = 0
minVol = -65
maxVol = 0
val = 0
detector = htm.handDetector(detectionCon=0.7)

# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(
#     IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = cast(interface, POINTER(IAudioEndpointVolume))
#  volume.GetMute()
#  volume.GetMasterVolumeLevel()
# print(volume.GetVolumeRange())
# volume.SetMasterVolumeLevel(-20.0, None)

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]     
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1 + y2)//2

        cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)
        #* Hand range 18 - 120
        #* Volume Range -65 - 0
        vol = np.interp(length, [18, 120], [minVol, maxVol])
        volBar = np.interp(length, [18, 120], [380, 108])
        volume = np.interp(length, [18, 120], [0, 100])
        print(vol)

        if length <= 18:
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        if length >= 120:
            cv2.circle(img, (cx, cy), 5, (0,0,255), cv2.FILLED)  

        cv2.rectangle(img, (560, 108), (590, 380), (0, 255, 0), 3)
        cv2.rectangle(img, (560, int(volBar)), (590, 380), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volume)}%',(545, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 1)

    cv2.imshow("Video", img)
    cv2.waitKey(1)
