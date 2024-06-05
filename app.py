import streamlit as st
import cv2
from deepface import DeepFace
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

perry_image_path = "image.jpg"

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.perry_detected = False
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        result = DeepFace.verify(img1_path=perry_image_path, img2_path=img, enforce_detection=False, model_name="VGG-Face")

        if result['verified'] == False:
            face = result['facial_areas']["img2"]
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        if result['verified']:
            self.perry_detected = True
            face = result['facial_areas']["img2"]
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Happy 22nd Birthday Perry", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2) 
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #faebd7;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">AIV System</h1>', unsafe_allow_html=True)

webrtc_ctx = webrtc_streamer(
    key="face_detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_transformer_factory=FaceDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    #async_processing=True,
)


if webrtc_ctx.video_transformer:
    #if webrtc_ctx.video_transformer.perry_detected:
    st.success("Please press the button below, when your identity is verified")
    st.link_button("Press", "https://drive.google.com/file/d/19mlhZ54PrIlBka5bxS3c9jWahxmI0NcV/view?usp=sharing")
