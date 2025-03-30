import cv2

frameWidth=320
frameHeight=240
minArea=500
numPlateCascade=cv2.CascadeClassifier('Resources/haarcascade_russian_plate_number.xml')
color=(255,0,255)
count=0

cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,100)

while True:
    success,img=cap.read()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    numberPlates=numPlateCascade.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in numberPlates:
        area=w*h
        if area>minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,"Number Plate",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
            imgROI=img[y:y+h,x:x+w]
            cv2.imshow('ROI',imgROI)


    cv2.imshow('output',img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Resources/Scanned/No_Plate_"+str(count)+".jpg",imgROI)
        count+=1
        break
