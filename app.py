import streamlit as st
import cv2
import face_recognition
import numpy as np

# Load Perry's reference image and encode it
perry_image = face_recognition.load_image_file("image.jpg")
perry_face_encoding = face_recognition.face_encodings(perry_image)[0]

# Create a list of known face encodings and their names
known_face_encodings = [perry_face_encoding]
known_face_names = ["Perry"]

def process_frame(frame):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Check if the face is a match for Perry's face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a frame around Perry's face and display the birthday message
        if name == "Perry":
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Happy 22nd Birthday Perry!", (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)

    return frame

st.title("Happy Birthday Perry!")
run = st.checkbox('Run')

# Create a video capture object
video_capture = cv2.VideoCapture(0)

frame_window = st.image([])

while run:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Process the frame
    frame = process_frame(frame)

    # Display the frame in Streamlit
    frame_window.image(frame, channels='BGR')

# Release the video capture object
video_capture.release()
