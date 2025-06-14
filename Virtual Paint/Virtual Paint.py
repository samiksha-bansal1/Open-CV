import cv2
import numpy as np

#webcam
frameWidth=640
frameHeight=480

cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,150)

#orange purple green blue
myColors=[[5,107,0,19,255,255],[133,56,0,159,156,255],[57,76,0,100,255,255],[90,48,0,118,255,255]]
myColorsValues=[[51,153,255],[255,0,255],[0,255,0],[255,0,0]]
myPoints=[]
# x,y,colorId

#color detection
def findColor(img,myColors,myColorsValues):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    count=0
    newPoints=[]
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x,y=getContours(mask)
        cv2.circle(imgResult,(x,y),10,myColorsValues[count],cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count+=1
    return newPoints

       # cv2.imshow(str(color[0]),mask)


#finding contours
def getContours(img):
    x,y,w,h=0,0,0,0
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>500:
            # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), cv2.FILLED)
            peri=cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h=cv2.boundingRect(approx)
            return x+w//2,y
    return 0,0

#drawing
def drawOnCanvas(myPoints,myColorsValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorsValues[point[2]], cv2.FILLED)




while True:
    success,img=cap.read()
    imgResult=img.copy()
    newPoints=findColor(img,myColors,myColorsValues)
    if len(newPoints)>0:
        for newP in newPoints:
            myPoints.append(newP)

    if len(myPoints)!=0:
        drawOnCanvas(myPoints,myColorsValues)

    cv2.imshow('Result',imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


