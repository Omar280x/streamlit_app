import streamlit as st
import cv2
from deepface import DeepFace

perry_image_path = "image.jpg"

def detect_faces_and_stream():
    # OpenCV Video Capture
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        st.error("Error: Unable to open webcam.")
        return

    # Display success message
    st.success("Webcam is now open. Press 'q' to exit.")

    # Create a window to display the webcam stream
    cv2.namedWindow("AIV System", cv2.WINDOW_NORMAL)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Perform face detection
            result = DeepFace.verify(img1_path=perry_image_path, img2_path=frame, enforce_detection=False, model_name="Facenet")

            if result['verified']:
                face = result['facial_areas']['img2']
                x, y, w, h = face['x'], face['y'], face['w'], face['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Happy 22nd Birthday Perry", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

            # Display the frame
            st.image(frame, channels="BGR")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Streamlit App
st.title("AIV System")

st.write("This app streams from your webcam and performs face detection.")

detect_faces_and_stream()
