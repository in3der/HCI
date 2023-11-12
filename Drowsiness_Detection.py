from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

import time
import atexit
import subprocess
import sys
import psutil

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0



while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				print ("Drowsy")
				## 졸림 신호 들어오면 iris_test2.py 실행 --- 아직 고쳐야한다. import iris_test하고 threading하면 순식간에 꺼졌다가 사라짐. 
				# a.py 프로세스 종료 함수
				def terminate_a_py():
					for process in psutil.process_iter(['pid', 'name']):
						if 'python' in process.info['name']:
							pid = process.info['pid']
							subprocess.run(["taskkill", "/F", "/PID", str(pid)])  # Windows에서의 프로세스 종료 명령어
							print(f"Process (PID: {pid}) terminated.")
							return

                # a.py 종료 후 b.py 실행
				terminate_a_py()
				#with open('iris_test2.py', 'r', encoding='utf-8') as f:
				#with open('iris_test2.py', 'r', encoding='latin-1') as f:
				#	code = compile(f.read(), 'iris_test2.py', 'exec')
				#	exec(code)
				with open('iris_test2.py', 'rb') as f:
					print("drowsy123")
					code = compile(f.read(), 'iris_test2.py', 'exec')
					exec(code)

                
				print("drowsy1111")
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release() 
