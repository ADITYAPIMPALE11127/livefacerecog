Project Title: Face Recognition System

Description:
This project implements a real-time face recognition system using computer vision techniques and deep learning models. 
The system captures video input from a webcam, detects faces in the frames, and matches them against a reference image using the DeepFace library. 
Upon detection, it provides visual feedback indicating whether a match is found or not.

Key Features:

Real-time face detection and recognition.
Utilizes DeepFace library for accurate face verification.
Multithreaded processing for improved performance.
Adjustable frame resolution for flexibility.
Easily extendable for integration into various applications.
Usage:

Install the required dependencies listed in the requirements.txt file.
Place the reference image (the image to be matched against) in the project directory.
Run the face_recognition.py script to start the face recognition system.
Press 'q' to quit the application.
Dependencies:

OpenCV
DeepFace
Python threading module
