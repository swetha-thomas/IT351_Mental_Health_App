
import csv
import requests
import re
from unittest import result
from urllib import request
import cv2
from django.contrib.auth.decorators import login_required
import pandas as pd


# # Create your views here.
from django.shortcuts import redirect, render
from django.template import Context

from recognition.forms import usernameForm

import numpy as np
import cv2
import pyaudio
import wave

def home(request):

	return render(request, 'home.html')

def question1(request):
	return render(request, 'question1.html')

def question2(request):
	return render(request, 'question2.html')

def question3(request):
	return render(request, 'question3.html')

def question4(request):
	return render(request, 'question4.html')

def question5(request):
	return render(request, 'question5.html')

def answer1(request):

	# i= request.POST['i']
	
	# audio=pyaudio.PyAudio()
	# stream = audio.open(format=pyaudio.paInt16, channels=1, rate = 44100, input=True,frames_per_buffer=1024)
	# frames = []
	# key=cv2.waitKey(1)
	# i=1
	# while True:
	# 		data = stream.read(1024)
	# 		frames.append(data)
	# 		if i==0:
	# 			break

		

	# stream.stop_stream()
	# stream.close()
	# audio.terminate()

	# sound_file=wave.open("AudioOutput.wav","wb")
	# sound_file.setnchannels(1)
	# sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
	# sound_file.setframerate(44100)
	# sound_file.writeframes(b''.join(frames))
	# sound_file.close()


	chunk = 1024  # Record in chunks of 1024 samples
	sample_format = pyaudio.paInt16  # 16 bits per sample
	channels = 2
	fs = 44100  # Record at 44100 samples per second
	seconds = 60
	filename = "output.wav"

	p = pyaudio.PyAudio()  # Create an interface to PortAudio

	print('Recording')

	stream = p.open(format=sample_format,
					channels=channels,
					rate=fs,
					frames_per_buffer=chunk,
					input=True)

	frames = []  # Initialize array to store frames

	# Store data in chunks for 3 seconds
	for i in range(0, int(fs / chunk * seconds)):
		data = stream.read(chunk)
		frames.append(data)

	# Stop and close the stream 
	stream.stop_stream()
	stream.close()
	# Terminate the PortAudio interface
	p.terminate()

	print('Finished recording')

	# Save the recorded data as a WAV file
	wf = wave.open(filename, 'wb')
	wf.setnchannels(channels)
	wf.setsampwidth(p.get_sample_size(sample_format))
	wf.setframerate(fs)
	wf.writeframes(b''.join(frames))
	wf.close()


	return render(request,'question2.html')


import cv2 
def end():
	r= requests.post('answer1', params={'i':0}, )

	return render(request, 'results.html')


def clickPicture1(request):
	key = cv2. waitKey(1)
	webcam = cv2.VideoCapture(0)
	# user_name = request.POST['user_name']
	i=0
	while True:
		i+=1
		try:
			check, frame = webcam.read()
			# print(check) #prints true as long as the webcam is running
			# print(frame) #prints matrix values of each framecd 
			cv2.imshow("Capturing", frame)
			key = cv2.waitKey(1)
			if i==5: 
				image_path = 'recognition/ImagesAttendance/img1.jpg'
				
				cv2.imwrite(filename=image_path, img=frame)
				webcam.release()
			
				img_new = cv2.imread(image_path)
				img_new = cv2.imshow("Captured Image", img_new)
				cv2.waitKey(1650)
				cv2.destroyAllWindows()
				break
			elif key == ord('q'):
				print("Turning off camera.")
				webcam.release()
				print("Camera off.")
				print("Program ended.")
				cv2.destroyAllWindows()
				break
			
		except(KeyboardInterrupt):
			print("Turning off camera.")
			webcam.release()
			print("Camera off.")
			print("Program ended.")
			cv2.destroyAllWindows()
			break
		
	return render(request,'question2.html')

def clickPicture2(request):
	key = cv2. waitKey(1)
	webcam = cv2.VideoCapture(0)
	# user_name = request.POST['user_name']
	i=0
	while True:
		i+=1
		try:
			check, frame = webcam.read()
			# print(check) #prints true as long as the webcam is running
			# print(frame) #prints matrix values of each framecd 
			cv2.imshow("Capturing", frame)
			key = cv2.waitKey(1)
			if i==5: 
				image_path = 'recognition/ImagesAttendance/img2.jpg'
				
				cv2.imwrite(filename=image_path, img=frame)
				webcam.release()
			
				img_new = cv2.imread(image_path)
				img_new = cv2.imshow("Captured Image", img_new)
				cv2.waitKey(1650)
				cv2.destroyAllWindows()
				break
			elif key == ord('q'):
				print("Turning off camera.")
				webcam.release()
				print("Camera off.")
				print("Program ended.")
				cv2.destroyAllWindows()
				break
			
		except(KeyboardInterrupt):
			print("Turning off camera.")
			webcam.release()
			print("Camera off.")
			print("Program ended.")
			cv2.destroyAllWindows()
			break
		
	return render(request,'question3.html')


def clickPicture3(request):
	key = cv2. waitKey(1)
	webcam = cv2.VideoCapture(0)
	# user_name = request.POST['user_name']
	i=0
	while True:
		i+=1
		try:
			check, frame = webcam.read()
			# print(check) #prints true as long as the webcam is running
			# print(frame) #prints matrix values of each framecd 
			cv2.imshow("Capturing", frame)
			key = cv2.waitKey(1)
			if i==5: 
				image_path = 'recognition/ImagesAttendance/img3.jpg'
				
				cv2.imwrite(filename=image_path, img=frame)
				webcam.release()
			
				img_new = cv2.imread(image_path)
				img_new = cv2.imshow("Captured Image", img_new)
				cv2.waitKey(1650)
				cv2.destroyAllWindows()
				break
			elif key == ord('q'):
				print("Turning off camera.")
				webcam.release()
				print("Camera off.")
				print("Program ended.")
				cv2.destroyAllWindows()
				break
			
		except(KeyboardInterrupt):
			print("Turning off camera.")
			webcam.release()
			print("Camera off.")
			print("Program ended.")
			cv2.destroyAllWindows()
			break
		
	return render(request,'question4.html')



def clickPicture4(request):
	key = cv2. waitKey(1)
	webcam = cv2.VideoCapture(0)
	# user_name = request.POST['user_name']
	i=0
	while True:
		i+=1
		try:
			check, frame = webcam.read()
			# print(check) #prints true as long as the webcam is running
			# print(frame) #prints matrix values of each framecd 
			cv2.imshow("Capturing", frame)
			key = cv2.waitKey(1)
			if i==5: 
				image_path = 'recognition/ImagesAttendance/img4.jpg'
				
				cv2.imwrite(filename=image_path, img=frame)
				webcam.release()
			
				img_new = cv2.imread(image_path)
				img_new = cv2.imshow("Captured Image", img_new)
				cv2.waitKey(1650)
				cv2.destroyAllWindows()
				break
			elif key == ord('q'):
				print("Turning off camera.")
				webcam.release()
				print("Camera off.")
				print("Program ended.")
				cv2.destroyAllWindows()
				break
			
		except(KeyboardInterrupt):
			print("Turning off camera.")
			webcam.release()
			print("Camera off.")
			print("Program ended.")
			cv2.destroyAllWindows()
			break
		
	return render(request,'question5.html')

def clickPicture5(request):
	key = cv2. waitKey(1)
	webcam = cv2.VideoCapture(0)
	# user_name = request.POST['user_name']
	i=0
	while True:
		i+=1
		try:
			check, frame = webcam.read()
			# print(check) #prints true as long as the webcam is running
			# print(frame) #prints matrix values of each framecd 
			cv2.imshow("Capturing", frame)
			key = cv2.waitKey(1)
			if i==5: 
				image_path = 'recognition/ImagesAttendance/img5.jpg'
				
				cv2.imwrite(filename=image_path, img=frame)
				webcam.release()
			
				img_new = cv2.imread(image_path)
				img_new = cv2.imshow("Captured Image", img_new)
				cv2.waitKey(1650)
				cv2.destroyAllWindows()
				break
			elif key == ord('q'):
				print("Turning off camera.")
				webcam.release()
				print("Camera off.")
				print("Program ended.")
				cv2.destroyAllWindows()
				break
			
		except(KeyboardInterrupt):
			print("Turning off camera.")
			webcam.release()
			print("Camera off.")
			print("Program ended.")
			cv2.destroyAllWindows()
			break
		
	return render(request,'results.html')



#!/usr/bin/env python
                              # Importing required modules
import cv2                                   # OpenCV # For face emotion detection
from fer import FER                          # https://pypi.org/project/fer/
import matplotlib.pyplot as plt              # MatPlotLib for plots
from moviepy.editor import VideoFileClip     # Importing from MoviePy and datetime
import datetime
import pandas as pd                          
import numpy as np
import seaborn as sns                        # Seaborn for Heat-Maps
import os                                    # for Speech to text: imports
import subprocess
import speech_recognition as sr
import ffmpeg
import shlex
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_video():
    
    ''' getting video file from directory'''
    
    myvideo = settings.video_file
    return myvideo


def emotions_face_video(myvideo):
    
    ''' Functions to return dictonary of bounding
    boxes for faces, emotions and scores'''
    
    clip = VideoFileClip(myvideo)
    duration = clip.duration                         
    vidcap = cv2.VideoCapture(myvideo)                      # VideCapture from cv2
    i = 0                                                   # initiate the variable for loop, will run for number of frames/images
    d = []                                                  # dictionary to capture the input of each image
    sec = 0                                                 # Variable to capture frame at particular time in the video
    frameRate = 1.0                                         # frameRate, to alter the time at which the frame is captured
    while i < abs((duration/frameRate) + 1):                # Numebr of frames based on duration and frameRate
            sec = sec + frameRate
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)     #Capturing video at particular intervals
            ret, image = vidcap.read() 
            if ret:                                         # If it has a frame
                    cv2.imwrite("image.jpg", image)         # saving image
                    img = plt.imread("image.jpg")           # reading image
                    detector = FER()                        # Calling fer for using already trained model
                    d = d + detector.detect_emotions(img)   # dictionary to store output of each image
            i = i + 1                                       # incrementing Loop
    return d



def emotion_face_video_dataframe(d):
    
    
    ''' Sentiment Analysis based on emotion detection 
      returns list of dictionaries for each image emotions and scores '''
    
     
    m = len(d)                                                                    # Get length of the dictinary
    cols =['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']     # Based on Trained Model in Fer
    df = pd.DataFrame(columns = cols)                                             # initiate empty dataframe
    list_dict = []                                                                # initate empty dictionary
    for i in range(0,m): 
        list_dict.append(d[i]['emotions'])                                  # Appending data from nested dictionary for each image in a list
    df= df.append(list_dict)
    print(df)                                                # Appending data from list into a empty dataframe
    neg = -(df['angry'][0]+df['fear'][0]+df['sad'][0])
    pos = df['happy'][0]+df['surprise'][0]
    score = neg+pos
    # print(score)
    return score



def speech_to_text():
    
    '''converts speech to text using recognizer from Google  '''

    # command = "ffmpeg -i "  + myvideo +  " Test4.mp3"                       # Command line to convert mp4 file into mp3
    # args = shlex.split(command)                                             # Split the args as required by subprocess
    # subprocess.run(args)  

    # command = "ffmpeg -i Test4.mp3  Test4.wav"                              # Command line to convert to mp3 into wav file
    # args = shlex.split(command)
    # subprocess.run(args)

    r = sr.Recognizer()                                                     # Making an instance of Recognizer
    with sr.AudioFile('output.wav') as source:               
        audio = r.record(source, duration=50)                               # duration 100 secs
        try:
            text_output = r.recognize_google(audio, language='en-IN')
        except Exception as e:
            print("could not understand audio")                             # Exception for empty audio or other lanaguage
    return text_output
    

def sentiment_Analysis_text(text_output):
    
    ''' Sentiment Analysis on the text'''
    
    nltk.download('vader_lexicon')                                   # https://www.kaggle.com/nltkdata/vader-lexicon
                                                                     # VADER Sentiment Analysis. 
                                                                     # VADER (Valence Aware Dictionary and sentiment Reasoner) is a lexicon 
    senti = SentimentIntensityAnalyzer()                             # and rule-based sentiment analysis tool 
    senti_text = senti.polarity_scores(text_output)                  # Give {'neg': 0.083, 'neu': 0.565, 'pos': 0.352, 'compound': 0.8733}
                                                                  
    stopwords = ["a", "this", "is", "and", "i", "to", "for", "very",                 # Excluding stopwords from text output(text to speech)
                 "know", "all", "the", "here", "about", "people", "you", "that"]
    
    reduced = list(filter(lambda w: w not in stopwords, (text_output.lower()).split()))
    
    data =({
    "Words":["Paragraph"] + reduced,
    "Sentiment":[senti_text["compound"]] + [senti.polarity_scores(word)["compound"] 
                                            for word in reduced]
     }) 
    
    score = senti_text["compound"]
    return score, data


# def video_sentiments(df_faces):
    
#     '''Using Seaborn to show heatmap of emotions with their probablities for video '''

#     fig, ax = plt.subplots(figsize=(10,10)) 
#     sns.heatmap(df_faces, annot=True)
   
#     return None

   
def text_sentiments(data):
    
    ''' returns heatmap of text sentiments for text'''
    
    grid_kws = {"height_ratios": (0.1, 0.007), "hspace": 2}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    f.set_figwidth(20)
    f.set_figheight(3)
    sns.heatmap(pd.DataFrame(data).set_index("Words").T,center=0, ax=ax, 
                annot=True, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"}, cmap = "PiYG")
    


def calculating(request):
	print("hello")
	
	# myvideo = get_video()                                         # Small video 40-60 secs
	# d = emotions_face_video(myvideo)                              # Calling emotions_face_video, returns list of emotion- score dictionary 
	# face_score = emotion_face_video_dataframe(d)                          # Dictionary to dataframe
	# video_sentiments(df)                                          # Heat Map


	text_output = speech_to_text()                         # Converting speech to text
	print(text_output)          
	                                
	# text_score, data = sentiment_Analysis_text(text_output)       #score and data for heatmap
	# print(text_score)                                             # overall positive, negative score
	# text_sentiments(data)                                         # Heat Map
	# # total = face_score +text_score
	# total = text_score
	# print(total)
	# if total<-0.05:
	# 	print("sad")
	# elif total<0.05:
	# 	print("neutral")
	# else:
	# 	print("happy")
	
	return render(request,'results.html')

























































@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'admin_dashboard.html')
	else:
		print("not admin")

		return render(request,'employee_dashboard.html')



import face_recognition
import os
from datetime import datetime
from datetime import date
import numpy as np

def findEncodings(images):
		encodeList=[]

		for img in images:
			img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

			if len(face_recognition.face_encodings(img2))!=0:
				encode = face_recognition.face_encodings(img2)[0]
				encodeList.append(encode)
			print(encodeList)
		return encodeList



def markAttendance(name):
		with open('recognition\AttendanceIn.csv','r+') as f:
			myDataList = f.readlines()
			nameList = []
			for line in myDataList:
				entry = line.split(',')
				nameList.append(entry[0])
				print(nameList)
            
			now = datetime.now()
			dt = date.today()
			dateString= dt.strftime('%d/%m/%Y')
			dtString = now.strftime('%H:%M:%S')
			print(dateString)
			f.writelines(f'\n{name},{dateString},{dtString}')

def markAttendanceOut(name):
		with open('recognition\AttendanceOut.csv','r+') as f:
			myDataList = f.readlines()
			nameList = []
			for line in myDataList:
				entry = line.split(',')
				nameList.append(entry[0])
            
			now = datetime.now()
			dt = date.today()
			dateString= dt.strftime('%d/%m/%Y')
			dtString = now.strftime('%H:%M:%S')
			print(dateString)
			f.writelines(f'\n{name},{dateString},{dtString}')

def mark_your_attendance(request):
	flagin=1
	count=0
	path = 'recognition\ImagesAttendance'
	images = []
	classNames = []
	myList = os.listdir(path)
	print(myList)

    
	for cl in myList:
		curImg = cv2.imread(f'{path}/{cl}')
		images.append(curImg)
		classNames.append(os.path.splitext(cl)[0])

	print(classNames)

	encodeListKnown = findEncodings(images)
	print("Encoding complete")

	cap = cv2.VideoCapture(0)

	while flagin==1 and count<3:
		count+=1
		success,img = cap.read()
		imgS = cv2.resize(img,(0,0),None,0.25,0.25)
		imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

		facesCurFrame = face_recognition.face_locations(imgS)
		encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

		for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
			matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
			faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
			#print(faceDis)
			matchIndex = np.argmin(faceDis)

			if matches[matchIndex]:
				
				name = classNames[matchIndex].upper()
				#print(name)
				y1,x2,y2,x1 = faceLoc
				y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
				cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
				cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
				cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
				cv2.putText(img,'Successful!',(200,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
				# cv2.waitKey(1)
				
				if flagin==1:
					markAttendance(name)
					flagin=0

			else: 
				y1,x2,y2,x1 = faceLoc
				y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
				cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
				cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
				cv2.putText(img,'FAIL',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
				cv2.putText(img,'Attempt:'+str(count),(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

				cv2.putText(img,'Unregisterd User',(200,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
				cv2.waitKey(5)
				if count>=3 and flagin==1:
					flagin=0
				

			

		cv2.imshow('Webcam',img)
		cv2.waitKey(2500)  
		

	return redirect('home')





def mark_your_attendance_out(request):
	flagout=1
	count=1

	
	path = 'recognition\ImagesAttendance'
	images = []
	classNames = []
	myList = os.listdir(path)
	print(myList)


	for cl in myList:
		curImg = cv2.imread(f'{path}/{cl}')
		images.append(curImg)
		classNames.append(os.path.splitext(cl)[0])

	print(classNames)

	encodeListKnown = findEncodings(images)
	print("Encoding complete")

	cap = cv2.VideoCapture(0)

	while flagout==1 and count<3:
		count+=1
		success,img = cap.read()
		imgS = cv2.resize(img,(0,0),None,0.25,0.25)
		imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

		facesCurFrame = face_recognition.face_locations(imgS)
		encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

		for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
			matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
			faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
			#print(faceDis)
			matchIndex = np.argmin(faceDis)

			if matches[matchIndex]:
				
				name = classNames[matchIndex].upper()
				#print(name)
				y1,x2,y2,x1 = faceLoc
				y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
				cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
				cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
				cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
				cv2.putText(img,'Successful!',(200,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
				# cv2.waitKey(1)
				
				if flagout==1:
					markAttendanceOut(name)
					flagout=0

			else: 
				y1,x2,y2,x1 = faceLoc
				y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
				cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
				cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
				cv2.putText(img,'FAIL',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
				cv2.putText(img,'Attempt:'+str(count),(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

				cv2.putText(img,'Unregisterd User',(200,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
				cv2.waitKey(5)
				if count>=3 and flagout==1:
					flagout=0
				

			

		cv2.imshow('Webcam',img)
		cv2.waitKey(2500)
		

	return redirect('home')

    
# @login_required
# def not_authorised(request):
# 	return render(request,'not_authorised.html')



# @login_required
def view_attendance_in(request):
	csv_fp = pd.read_csv('recognition\AttendanceIn.csv',header=0)
	context2 = csv_fp.to_dict('list')
	result_set = dict()
	result_set['data'] = context2
	result_set['header'] =  [v for v in context2.keys()]
	return render(request, 'view_attendance_in.html', result_set)
	
	
def view_attendance_home(request):

	csv_fp = pd.read_csv('recognition\AttendanceOut.csv',header=0)
	context2 = csv_fp.to_dict('list')
	result_set = dict()
	result_set['data'] = context2
	result_set['header'] =  [v for v in context2.keys()]
	return render(request, 'view_attendance_home.html', result_set)
	
    
	