import streamlit as st
import cv2
from deepface import DeepFace
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

perry_image_path = "image.jpg"

class FaceDetectionTransformer:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        result = DeepFace.verify(img1_path=perry_image_path, img2_path=img, enforce_detection=False)

        if result['verified'] == False:
            face = result['region']
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        if result['verified']:
            face = result['region']
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Happy 22nd Birthday Perry", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2) 

            return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("AIV System")

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_transformer_factory=FaceDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# webrtc_ctx = webrtc_streamer(
#     key="example",
#     #mode=WebRtcMode.SENDRECV,
#     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     video_processor_factory=FaceDetectionTransformer,
#     #media_stream_constraints={"video": True, "audio": False},
# )


    # if webrtc_ctx.video_processor:
    #     # while not webrtc_ctx.video_processor.face_detected and time.time() - webrtc_ctx.video_processor.start_time < detection_duration:
    #     #     pass
    
    #     if webrtc_ctx.video_processor.face_detected:
    #         st.success("Perry's identity is verified!, Click the link below.")
    #         with open("present.rar", "rb") as file:
    #             st.download_button(
    #                 label="Download present",
    #                 data=file,
    #                 file_name="archive.rar",
    #                 mime="application/x-rar-compressed"
    #             )
    #     else:
    #         st.warning("Perry not detected within 15 seconds.")
