import threading  # Importing threading module for concurrent execution
import cv2  # Importing OpenCV library for computer vision tasks
from deepface import DeepFace  # Importing DeepFace library for face recognition

# Initializing camera capture object
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Setting capture properties for frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0  # Counter for frame processing

face_match = False  # Flag to indicate if a face match is found
face_match_lock = threading.Lock()  # Lock to synchronize access to face_match variable

# Loading reference image for face verification
reference_img = cv2.imread("reference.jpg")


# Function to check if a face is detected in the frame and verify it against a reference image
def check_face(frame1):
    global face_match
    try:
        # Using DeepFace library to verify the face in the frame against the reference image
        verified = DeepFace.verify(frame1, reference_img.copy())['verified']
        # Updating face_match variable with the verification result
        with face_match_lock:
            face_match = verified
    except ValueError:
        # Handling exceptions, setting face_match to False if an error occurs
        with face_match_lock:
            face_match = False


# Main loop for capturing and processing frames
while True:
    ret, frame = cap.read()  # Reading a frame from the camera

    if ret:
        if counter % 30 == 0:  # Processing every 30th frame
            try:
                # Creating a new thread to check for face match asynchronously
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1  # Incrementing the counter

        # Drawing text on the frame based on the face match status
        with face_match_lock:
            if not face_match:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("video", frame)  # Displaying the processed frame
    key = cv2.waitKey(1)
    if key == ord("q"):  # Exiting the loop if 'q' key is pressed
        break

cv2.destroyAllWindows()  # Closing all OpenCV windows
cap.release()  # Releasing the camera capture resources
