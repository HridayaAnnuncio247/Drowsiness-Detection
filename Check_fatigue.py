import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import time
from utils import utils # used for yield

class fatigue:

	def __init__(self):
		self.utils = utils() # for yield
		self.thld = 0.0
		self.detector = None
		self.predictor = None


	def load_models(self):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



				
	def get_landmarks(self,frame):

		#print("inside get_landmarks")

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = self.detector(gray)
		for face in faces:
			x1 = face.left()
			y1 = face.top()
			x2 = face.right()
			y2 = face.bottom()
			        
			t_g_s=20
			c_l=2.5
					
			clahe = cv2.createCLAHE(clipLimit=c_l, tileGridSize=(t_g_s,t_g_s))
			adjusted = clahe.apply(gray[y1-100:y2+100,x1-100:x2+100])
			#adjusted=gray[y1-100:y2+100,x1-100:x2+100]
					
			faces2 = self.detector(adjusted)
			for face2 in faces2:
				#print("entered faces2")
				x1 = face.left()
				y1 = face.top()
				x2 = face.right()
				y2 = face.bottom()
				landmarks = self.predictor(adjusted, face2)

				return landmarks
						

	def get_threshold(self,vid):

		#print("inside get_threshold")
		print(vid)
		cap = self.utils.vid_capture(vid)
		print("after")
		ear = []

		tm_1=time.time()
		f=0

		while cap.isOpened(): # While the camera is open

			ret, frame = cap.read()
		    
			if ret:	
				f=f+1	   

				landmarks=self.get_landmarks(frame)
				if landmarks:
					leftEAR = self.utils.eye_aspect_ratio([landmarks.part(36), 
			                                            landmarks.part(37), 
			                                            landmarks.part(38), 
			                                            landmarks.part(39), 
			                                            landmarks.part(40), 
			                                            landmarks.part(41)])
					rightEAR = self.utils.eye_aspect_ratio([landmarks.part(42), 
			                                            landmarks.part(43), 
			                                            landmarks.part(44), 
			                                            landmarks.part(45), 
			                                            landmarks.part(46), 
			                                            landmarks.part(47)])
			                

					avg = (leftEAR+rightEAR)/2.0
					ear.append(avg)
			        
				key = cv2.waitKey(1)#1 means that a frame in the video will be held for atleast 1 ms and then changed to next frame in streaming video. How much more time it takes greater than a ms depends on other instruction being taken care of in the OS. 
				# 1 can be changed to other number based on how many ms you want the frame to wait.
				tm_2=time.time()
				cv2.imshow("Frame", frame)

				if key == 27 or (tm_2-tm_1 > 60) :# key ==27 implies that when the escape key is pressed, stop the video streaming.
					break # also breaks if there is more than a minute's gap since the camera was opened...I think that may not be required...Will have to check it out.
			else:
				break


		#print(f)
		cap.release()
		cv2.destroyAllWindows()


		self.thld = np.percentile(ear,10)# takes top 10th percentile of average distance so as to not include blinks


	def detect_fatigue(self,vid):
		st = time.time()
		print(st)

		
		#left = []
		#right = []
		#ear = []
		time_ms = []
		min_avg=0
		min_tm=0
		max_avg=0
		max_tm=0
		m_a_r = []
		time_ms_y = []
		min_avg_y=0
		min_tm_y=0
		max_avg_y=0
		max_tm_y=0

		pre = 0
		pre_y=0

		print("inside drowsiness_detection")

		#thld = self.get_threshold(vid)

		cap = self.utils.vid_capture(vid)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


		f=0
		fs = 0
		fs_m = 0
		fe = 0
		fe_m = 0

		while cap.isOpened():

			ret, frame = cap.read()

		    
			if ret:		   


				f=f+1
				print(f)
				if f>total_frames:
					break

				landmarks=self.get_landmarks(frame)
				if landmarks:
					leftEAR = self.utils.eye_aspect_ratio([landmarks.part(36), 
			                                            landmarks.part(37), 
			                                            landmarks.part(38), 
			                                            landmarks.part(39), 
			                                            landmarks.part(40), 
			                                            landmarks.part(41)])
					rightEAR = self.utils.eye_aspect_ratio([landmarks.part(42), 
			                                            landmarks.part(43), 
			                                            landmarks.part(44), 
			                                            landmarks.part(45), 
			                                            landmarks.part(46), 
			                                            landmarks.part(47)])
					mar = self.utils.mouth_aspect_ratio([landmarks.part(61), 
			                                        landmarks.part(67), 
			                                        landmarks.part(62), 
			                                        landmarks.part(66), 
			                                        landmarks.part(63), 
			                                        landmarks.part(65), 
			                                        landmarks.part(60), 
			                                        landmarks.part(64)])

					tm=time.time()*1000
					time_ms.append(tm)
					time_ms_y.append(tm)
			        #print(time.time())
			        
					#left.append(leftEAR)
					#right.append(rightEAR)
					avg = (leftEAR+rightEAR)/2.0
					#ear.append(avg)

					#DROWSINESS DETECTION
					if min_avg==0 :
			            #print("entered 1st if")
						min_avg=avg#setiing min avg for the first time
						min_tm = tm
						fs = f # blink start frame (s for start) ie.e the first time the EAR has gone 
					else:
						if (avg<(min_avg+0.03) and avg>(min_avg-0.03) and avg<self.thld):

							max_tm= tm
							fe = f
							if((max_tm-min_tm) >= 1000): # means if the interval of the blink going below average and going below threshold is more than 1000 ms, it could be drowsiness, cause a blonnk takes lesser time. Need to do more research on the amnt of time i takes for a blink.
								#print(max_tm-min_tm)
								#print("Drowsy behaviour")						
								#min_avg=0
								pre=1
								#print(min_tm/1000 , "-" , max_tm/1000)
								#print("frames",fs,"-",fe)
								#yield 1

						else:
							if pre==1:# drowsiness was detecteed in the previous frame but now in the else,avg has gone above min avg or threshold
								#min_avg=0
								print(min_tm/1000 , "-" , max_tm/1000)
								print("frames",fs,"-",fe)
								pre=0
								yield 1,fs,fe # 1 indicates that drowsiness has been detected. fs and fe used to talk about the number of frames used to detect drowsiness
							min_avg=avg
							min_tm = tm
							fs = f# reset start frame each time the EAR is not below min avg and threshold

					#YAWN DETECTION

					if min_avg_y==0 :
						min_avg_y=mar
						min_tm_y = tm
						fs_m = f
					else:
						#print("entered")
						if mar<(min_avg_y+0.05) and mar>(min_avg_y-0.05) and mar>0.27:
							max_tm_y= tm
							fe_m = f
							if((max_tm_y-min_tm_y) >= 1000):
								#print(max_tm_y-min_tm_y)
								#print("Yawning")
								#print(min_tm_y/1000 - st , "-" , max_tm_y/1000 - st)
								#print("frames",fs_m,"-",fe_m)
								pre_y=1
								#min_avg_y=0
								#yield 2

						else:
							if pre_y==1:
								print(max_tm_y-min_tm_y)
								print("frames",fs_m,"-",fe_m)
								#min_avg_y=0
								pre_y=0
								yield 2,fs,fe # 1 indicates that a yawn has been detected.


							min_avg_y=mar
							min_tm_y = tm
							fs_m = f


				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1)
				if key == 27:
					break
			else:
				break

		print("******************")
		print(f)
		cv2.destroyAllWindows()
		cap.release()




	'''def detect_drowsiness(self,avg):
		if min_avg==0 :
			            #print("entered 1st if")
			min_avg=avg
			min_tm = tm
		else:
			if avg<(min_avg+0.03) and avg>(min_avg-0.03) and avg<self.thld:

				max_tm= tm
				if((max_tm-min_tm) >= 1000):
					print(max_tm-min_tm)
					print("Drowsy behaviour")						
					min_avg=0
					return 1

					else:
						min_avg=avg
						min_tm = tm'''




'''
	def detect_yawn(self,vid):

		m_a_r = []
		time_ms = []
		min_avg=0
		min_tm=0
		max_avg=0
		max_tm=0
		

		cap = self.utils.vid_capture(vid)



		while cap.isOpened():

			ret, frame = cap.read()

		    
			if ret:		   

				landmarks=self.get_landmarks(frame)
				if landmarks:	
					mar = self.utils.mouth_aspect_ratio([landmarks.part(61), 
			                                        landmarks.part(67), 
			                                        landmarks.part(62), 
			                                        landmarks.part(66), 
			                                        landmarks.part(63), 
			                                        landmarks.part(65), 
			                                        landmarks.part(60), 
			                                        landmarks.part(64)])
					tm=time.time()*1000
					#time_ms.append(tm)
			        
					m_a_r.append(mar)

					if min_avg==0 :
						min_avg=mar
						min_tm = tm
					else:
						if mar<(min_avg+0.05) and mar>(min_avg-0.05) and mar>0.30:
							max_tm= tm
							if((max_tm-min_tm) >= 1000):
								print(max_tm-min_tm)
								print("Yawning")
								min_avg=0
								yield 2

						else:
							min_avg=mar
							min_tm = tm
				
				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1)
				if key == 27:
					break
			else:
				break


'''






