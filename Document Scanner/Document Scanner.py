import cv2
import numpy as np

frameWidth=640
frameHeight=480

widthIMG=640
heightIMG=480


# cap=cv2.VideoCapture(url)

cap=cv2.VideoCapture(0)

cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,100)

#preprocessing the image
def preprocessing(img):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny=cv2.Canny(imgBlur,100,200)

    kernel=np.ones((5,5),np.uint8)
    imgDial=cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres=cv2.erode(imgDial,kernel,iterations=1)

    return imgThres

#contours
def getContours(img):

    biggest=np.array([])
    maxArea=0

    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>500:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri=cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area>maxArea and len(approx)==4  :
                biggest=approx
                maxArea=area
            if biggest.size != 0:
                cv2.drawContours(imgContour, [biggest], -1, (255, 0, 0), 10)
    return biggest

#reorder
def reorder(myPoints):
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),np.int32)
    add=myPoints.sum(axis=1)
    # print(add)

    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[3]=myPoints[np.argmax(add)]
    diff=np.diff(myPoints,axis=1)
    myPointsNew[1]=myPoints[np.argmin(diff)]
    myPointsNew[2]=myPoints[np.argmax(diff)]

    return myPointsNew

def getWarp(img,biggest):
    if biggest.size == 0:
        print("No document detected!")
        return img

    biggest=reorder(biggest)

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthIMG, 0], [0, heightIMG], [widthIMG, heightIMG]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthIMG, heightIMG))

    imgCropped=imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped=cv2.resize(imgCropped,(widthIMG,heightIMG))
    return imgCropped

while True:
    sucess,img=cap.read()
    cv2.resize(img,(widthIMG,heightIMG))
    imgContour=img.copy()

    imgThres=preprocessing(img)
    biggest=getContours(imgThres)

    imgWarped=getWarp(img,biggest)

    cv2.imshow('frame',imgWarped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



