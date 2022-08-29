import cv2
from random import randrange

#load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#img to detect face
#img = cv2.imread('RDJ.jpg')
webcam = cv2.VideoCapture(0)

#iterate forever over frames
while True:
    #read the current frame
    #sucess_frame_read always returned true, we just want the frame for detection
    successful_frame_read, frame = webcam.read()
    

    #convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 2)

    cv2.imshow("Face Dectector", frame)
    key = cv2.waitKey(1)

    #stop if Q (ASCII 81 or 113) is pressed
    if key == 81 or key == 113:
        break

#release video capture object
webcam.release()
print("code complete")
