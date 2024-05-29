import streamlit as st
import cv2
import face_recognition
import numpy as np

perry_image = cv2.imread("image.jpg")

def process_frame(frame):
    frame = frame[:, :, ::-1]

    result = DeepFace.verify(img1_path = perry_image, img2_path = frame)

    verified_perry = result["verified"]

    if verified_perry:
        detected_face = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
        x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
    
        cv2.rectangle(frame, (x, y), (w+x, h+y), (0, 0, 255), 2)
        cv2.putText(frame, "Happy 22nd Birthday Perry", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)
    
        return frame

st.title("AIV System")
run = st.checkbox('Run')

video_capture = cv2.VideoCapture(0)

frame_window = st.image([])

while run:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = process_frame(frame)

    frame_window.image(frame, channels='BGR')

video_capture.release()

if verified_perry:
    st.success("Click the link below")
    with open(present_file, "rb") as file:
        btn = st.download_button(
            label="Download",
            data=file,
            file_name="perrylol99.rar",
            mime="application/x-rar-compressed")
