import streamlit as st
import cv2
from deepface import DeepFace
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

perry_image_path = "image.jpg"

perry_image = DeepFace.detectFace(img_path=perry_image_path)

face_detected = False
start_time = None
detection_duration = 15  

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_detected = False
        self.start_time = None

    def transform(self, frame):
        global face_detected, start_time

        img = frame.to_ndarray(format="bgr24")

        if self.start_time is None:
            self.start_time = time.time()

        try:
            detected_faces = DeepFace.detectFace(img, detector_backend='opencv', enforce_detection=False)

            if detected_faces.size != 0:
                result = DeepFace.verify(img1_path=perry_image_path, img2_path=detected_faces, enforce_detection=False)
                
                if result['verified']:
                    self.face_detected = True
                    # Draw a frame around Perry's face and display the birthday message
                    face = result['region']
                    x, y, w, h = face['x'], face['y'], face['w'], face['h']
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, "Happy 22nd Birthday Perry", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)

        except Exception as e:
            pass

        if time.time() - self.start_time > detection_duration:
            webrtc_streamer.key_stop()
        
        return img

st.title("AIV System")
run = st.button('Run')

# if run:
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=FaceDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.video_processor:
    while not webrtc_ctx.video_processor.face_detected and time.time() - webrtc_ctx.video_processor.start_time < detection_duration:
        pass

    if webrtc_ctx.video_processor.face_detected:
        st.success("Perry's identity is verified!, Click the link below.")
        with open("present.rar", "rb") as file:
            st.download_button(
                label="Download present",
                data=file,
                file_name="archive.rar",
                mime="application/x-rar-compressed"
            )
    else:
        st.warning("Perry not detected within 15 seconds.")
