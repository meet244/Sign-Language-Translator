import customtkinter as ctk
import csv
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
from model import KeyPointClassifier
import itertools
import copy
from datetime import datetime

# Function to calculate the landmark points from an image
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Iterate over each landmark and convert its coordinates
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# Function to preprocess landmark data
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# Load the KeyPointClassifier model
keypoint_classifier = KeyPointClassifier()

# Read labels from a CSV file
with open('model/keypoint_classifier/label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

# Set the appearance mode and color theme for the custom tkinter library
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Create the main window
window = ctk.CTk()
window.geometry('1080x1080')
window.title("HAND SIGNS")
prev = ""

# Function to open the camera and perform hand gesture recognition
def open_camera1():
    global prev
    width, height = 800, 600
    with mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5,static_image_mode=False) as hands:
            
            _, frame = vid.read()
            opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            opencv_image = cv2.resize(opencv_image, (width,height))
                        
            processFrames = hands.process(opencv_image)
            if processFrames.multi_hand_landmarks:
                for lm in processFrames.multi_hand_landmarks:
                    mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)

                    landmark_list = calc_landmark_list(frame, lm)

                    pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    cur = keypoint_classifier_labels[hand_sign_id]
                    if(cur == prev) : 
                        letter.configure(text=cur)
                    elif(cur):
                        prev = cur
                   
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.flip(frame,1)
            captured_image = Image.fromarray(frame)
            my_image = ctk.CTkImage(dark_image=captured_image,size=(340,335))
            video_lable.configure(image=my_image)
            video_lable.after(10, open_camera1)

# Initialize the video capture
vid = cv2.VideoCapture(0)
mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils
width, height = 600, 500

# Create the title label
i = 0
title = ctk.CTkFont(
     family='Consolas',
     weight='bold',
     size=25
)
Label = ctk.CTkLabel(
     window,
     text = 'HAND SIGNS',
     fg_color='steelblue',
     text_color= 'white',
     height= 40,
     font=title,
     corner_radius= 8)
Label.pack(side = ctk.TOP,fill=ctk.X,pady=(10,4),padx=(10,10))

# Create the main frame
main_frame = ctk.CTkFrame(master=window,
                          height=770,
                          corner_radius=8
                          )

main_frame.pack(fill = ctk.X , padx=(10,10),pady=(5,0))
MyFrame1=ctk.CTkFrame(master=main_frame,
                     height = 375,
                     width=365
                     )
MyFrame1.pack(fill = ctk.BOTH,expand=ctk.TRUE,side = ctk.LEFT,padx = (10,10),pady=(10,10))

# Create the video frame
video_frame = ctk.CTkFrame(master=MyFrame1,height=340,width=365,corner_radius=12)
video_frame.pack(side=ctk.TOP,fill=ctk.BOTH,expand = ctk.TRUE ,padx=(10,10),pady=(10,5))

# Create the video label
video_lable = ctk.CTkLabel(master=video_frame, text='',height=340,width=365,corner_radius=12)
video_lable.pack(fill=ctk.BOTH,padx=(0,0),pady=(0,0))

# Create a button to start the camera feed
Camera_feed_start= ctk.CTkButton(master=MyFrame1,text='START',height=40,width=250,border_width=0,corner_radius=12,command=lambda : open_camera1())
Camera_feed_start.pack(side = ctk.TOP,pady=(5,10))

MyFrame2=ctk.CTkFrame(master=main_frame,
                     height=375
                     ) 
MyFrame2.pack(fill = ctk.BOTH,side=ctk.LEFT,expand = ctk.TRUE,padx = (10,10),pady=(10,10))

# Create a font for displaying letters
myfont = ctk.CTkFont(
     family='Consolas',
     weight='bold',
     size=200
)
letter = ctk.CTkLabel(MyFrame2,
                          font=myfont,fg_color='#2B2B2B',justify=ctk.CENTER)
letter.pack(fill = ctk.BOTH,side=ctk.LEFT,expand = ctk.TRUE,padx = (10,10),pady=(10,10))
letter.configure(text='')

MyFrame3=ctk.CTkFrame(master=window,
                     height=175,
                     corner_radius=12
                     )
MyFrame3.pack(fill = ctk.X,expand = ctk.TRUE,padx = (10,10),pady=(10,10))

# Create a textbox for displaying a sentence
Sentence = ctk.CTkTextbox(MyFrame3,
                          font=("Consolas",24))
Sentence.pack(fill = ctk.X,side=ctk.LEFT,expand = ctk.TRUE,padx = (10,10),pady=(10,10))

# Start the tkinter main loop
window.mainloop()
