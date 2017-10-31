import cv2
from time import sleep
import numpy as np

car_cascade = cv2.CascadeClassifier('cars.xml')

cap = cv2.VideoCapture('video.avi')
f = 0
while True:
	f = f + 1
	ret, img = cap.read()
	#img = cv2.resize(img, (640, 480))
	i = 0
	if f > 500 :
		break
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cars = car_cascade.detectMultiScale(gray, 1.1, 1)
	for (x,y,w,h) in cars:
		roi = img[x:x+w, y:y+h]
		if len(roi) > 1 and len(roi[0]) > 1:
			hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			sensitivity = 25
    		lower_white = np.array([0,0,255-sensitivity])
    		upper_white = np.array([255,sensitivity,255])
    		mask= cv2.inRange(hsv, lower_white, upper_white)
    		res = cv2.bitwise_and(hsv, hsv, mask= mask)
    		average_color_per_row = np.average(res, axis=0)
    		avg_color = np.average(average_color_per_row, axis=0)
    		average_color = np.uint8(avg_color)
    		#print average_color
    		if average_color[2] + average_color[0] + average_color[1] > 0:
    			i = i + 1
		cv2.rectangle(img, (x+5,y+5), (x+w-5, y+h-5), (255,0,0), 2)
		font = cv2.FONT_HERSHEY_SIMPLEX
		sleep(0.01)
	cv2.putText(img, str("Cars Detected : " + str(len(cars))) ,(5,20), font, 0.4,(255,255,255),1,cv2.LINE_AA)
	cv2.putText(img, str("White Cars : " + str(i)) ,(5,35), font, 0.4,(255,255,255),1,cv2.LINE_AA)
	cv2.putText(img, str("Other Cars : " + str(len(cars) - i)) ,(5,50), font, 0.4,(255,255,255),1,cv2.LINE_AA)
	cv2.imshow('Car Detection', img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cap.release()
cv2.destroyAllWindows()
