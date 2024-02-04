import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import CNN
import torchvision.transforms as transforms
from inference import output
load_from_sys = True

if load_from_sys:
    hsv_value=np.array([[0,0,0],[255,255,255]])
   #hsv_value = np.load('hsv_value.npy',allow_pickle=True)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.uint8)

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

x2 = 0
y2 = 0

noise_thresh = 10

cv2.namedWindow('Trackbars')
cv2.createTrackbar("L - H", "Trackbars", 0, 179, lambda x: x)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, lambda x: x)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, lambda x: x)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, lambda x: x)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, lambda x: x)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, lambda x: x)

while True:
    shape = (3, 4, 5)
    roi= np.zeros(shape, dtype=np.float32)
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    #if canvas is not None:
      #  canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if load_from_sys:
        lower_range = hsv_value[0]
        upper_range = hsv_value[1]

    mask = cv2.inRange(hsv, lower_range, upper_range)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    x=np.zeros((28,28))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        #canvas2=cv2.drawContours(x,c,-1,(255,255,0),3)
        x1, y1, w, h = cv2.boundingRect(c)
        padding=30
        x1-=padding
        y1-=padding
        w+=2*padding    
        h+=2*padding
        x1=max(0,x1)
        y1=max(0,y1)

        cv2.rectangle(frame, (x1,y1), (x1+w,y1+h), (255, 0, 0), 2)
        roi=frame[y1:y1+h,x1:x1+w]
        roi=cv2.flip(roi,1)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi = cv2.inRange(roi_hsv, lower_range, upper_range)
        #roi=cv2.GaussianBlur(roi,(3,3),0)
        #roi=roi[:,:,2]
        #_,roi=cv2.threshold(roi,50,255,cv2.THRESH_BINARY)
        #roi=cv2.bitwise_not(roi)
        cv2.imshow('roi',roi)
        roi=cv2.resize(roi,(28,28),cv2.INTER_NEAREST)
        input_tensor=transforms.ToTensor()(roi)
        input_tensor = input_tensor.unsqueeze(0)
        scores=output(input_tensor)
        _,prediction=scores.max(1)
        text = f'Prediction: {prediction.item()}'
        position = (50, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_color = (255, 255, 255)  
        line_thickness = 2

        cv2.putText(frame, text, position, font, font_scale, font_color, line_thickness)
        print(prediction)
        

    #canvas = cv2.add(canvas, frame)

    stacked = np.hstack((frame,res))
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    #cv2.imshow('canvas2',canvas2)

    if cv2.waitKey(1) == 10:
        break


    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = np.zeros_like(frame)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    hsv_value = [lower_range, upper_range]
    #np.save('hsv_value.npy', hsv_value)

cv2.destroyAllWindows()
cap.release()