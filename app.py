import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# Load a pre-trained YOLOv8n model
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')  # Load the nano version of YOLOv8
    return model

model = load_model()

# Class names for COCO dataset (YOLO default)
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def draw_boxes(image, results):
    img_height, img_width, _ = image.shape
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        for box in boxes:
            # Get box coordinates in (left, top, right, bottom) format
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}"
            color = (0, 255, 0)  # Green color for bounding box

            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)
    annotated_img = draw_boxes(img.copy(), results)
    return annotated_img, results



def display_statistics(detected_classes):
    # detected_classes can be a list of class names (strings) or a list of results objects
    # If it's a list of results objects (from image or webcam frame-by-frame),
    # extract class names from there.
    # If it's already a list of class names (from video total statistics),
    # use it directly.
    if not detected_classes:
        st.write("No objects detected.")
        return

    if isinstance(detected_classes[0], str):
        # It's already a list of class names (for total video statistics)
        final_classes = detected_classes
    else:
        # It's a list of results objects (for image or frame-by-frame webcam)
        final_classes = []
        for r in detected_classes:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    final_classes.append(CLASS_NAMES[class_id])

    if not final_classes:
        st.write("No objects detected.")
        return

    class_counts = Counter(final_classes)
    with st.expander("Detected Objects", expanded=True):
        cols = st.columns(len(class_counts))
        for i, (class_name, count) in enumerate(class_counts.items()):
            with cols[i]:
                st.write(f"- {class_name}: {count}")

st.title("Real-time Object Detection and Annotation")

st.sidebar.header("Configuration")
image_width = st.sidebar.slider("Image Width", 100, 1500, 700, key="image_width_slider")
image_height = st.sidebar.slider("Image Height", 100, 1500, 500, key="image_height_slider")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, key="confidence_slider")

# Input options
input_option = st.sidebar.radio("Select Input Type", ("Image", "Video", "Webcam"), key="input_option_radio")

if input_option == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR", caption="Original Image", width=image_width)

        # Perform inference and draw bounding boxes
        annotated_image, results = predict_and_detect(model, image.copy(), conf=confidence_threshold)
        st.image(annotated_image, channels="BGR", caption="Detected Objects", width=image_width)
        display_statistics(results)

elif input_option == "Video":
    video_width = st.sidebar.slider("Video Width", 100, 1500, 700, key="video_width_slider")
    video_height = st.sidebar.slider("Video Height", 100, 1500, 500, key="video_height_slider")
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"]) 
    st.video(uploaded_file, format="video/mp4", start_time=0, loop=True, autoplay=True) # Removed width
        
        # Create a temporary file to save the uploaded video
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
        video_path = "temp_video.mp4"
        cap = cv2.VideoCapture(video_path)

        if "video_cap" not in st.session_state:
            st.session_state.video_cap = cv2.VideoCapture(video_path)
            st.session_state.frame_count = 0
            st.session_state.total_frames = int(st.session_state.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.session_state.video_processing_done = False

        cap = st.session_state.video_cap
        frame_count = st.session_state.frame_count
        total_frames = st.session_state.total_frames

        if not cap.isOpened():
            st.error("Error: Could not open video.")
            st.session_state.video_processing_done = True
        else:
            st.write("Processing video...")
            
            progress_bar = st.progress(frame_count / total_frames)
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()

            if not st.session_state.video_processing_done:
                ret, frame = cap.read()
                if ret:
                    # Perform inference and draw bounding boxes on the frame
                    annotated_frame, results = predict_and_detect(model, frame.copy(), conf=confidence_threshold)
                    
                    # Display the annotated frame
                    with frame_placeholder.container():
                        st.image(annotated_frame, channels="BGR", caption=f"Frame {frame_count+1}/{total_frames}", width=video_width)
                    
                    # Display statistics for the current frame
                    with stats_placeholder.container():
                        st.subheader(f"Statistics for Frame {frame_count+1}")
                        display_statistics(results)

                    st.session_state.frame_count += 1
                    progress_bar.progress(st.session_state.frame_count / total_frames)

                    if st.session_state.frame_count >= total_frames:
                        st.session_state.video_processing_done = True
                        st.success("Video processing complete!")
                        if 'video_cap' in st.session_state and st.session_state.video_cap is not None:
                            st.session_state.video_cap.release()
                            st.session_state.video_cap = None
                        st.rerun() # Rerun to trigger cleanup
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Next Frame", key="next_frame"):
                                st.rerun()
                        with col2:
                            if st.button("Stop Processing", key="stop_processing"):
                                st.session_state.video_processing_done = True
                                st.warning("Video processing stopped by user.")
                                if 'video_cap' in st.session_state and st.session_state.video_cap is not None:
                                    st.session_state.video_cap.release()
                                    st.session_state.video_cap = None
                                st.rerun() # Rerun to trigger cleanup
                else:
                    st.session_state.video_processing_done = True
                    st.success("Video processing complete!")
                    if 'video_cap' in st.session_state and st.session_state.video_cap is not None:
                        st.session_state.video_cap.release()
                        st.session_state.video_cap = None
                    st.rerun() # Rerun to trigger cleanup
            
            # Cleanup logic moved outside the main processing loop
            if st.session_state.video_processing_done:
                if 'video_cap' in st.session_state and st.session_state.video_cap is not None:
                    st.session_state.video_cap.release()
                    st.session_state.video_cap = None
                if os.path.exists("temp_video.mp4"):
                    os.remove("temp_video.mp4")
                st.success("Video processing complete!")
                # Reset session state for next video upload
                st.session_state.frame_count = 0
                st.session_state.total_frames = 0
                st.session_state.video_processing_done = False

elif input_option == "Webcam":
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False

    webcam_width = st.sidebar.slider("Webcam Width", 100, 1500, 700, key="webcam_width_slider")
    webcam_height = st.sidebar.slider("Webcam Height", 100, 1500, 500, key="webcam_height_slider")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Webcam"):
            st.session_state.webcam_running = True
            st.rerun()
    with col2:
        if st.button("Stop Webcam"):
            st.session_state.webcam_running = False
            if 'webcam_cap' in st.session_state and st.session_state.webcam_cap is not None:
                st.session_state.webcam_cap.release()
                st.session_state.webcam_cap = None
            st.rerun()

    if st.session_state.webcam_running:
        if 'webcam_cap' not in st.session_state or st.session_state.webcam_cap is None:
            st.session_state.webcam_cap = cv2.VideoCapture(0)  # 0 for default webcam
            if not st.session_state.webcam_cap.isOpened():
                st.error("Error: Could not open webcam.")
                st.session_state.webcam_running = False
                st.rerun()

        webcam_placeholder = st.empty()
        statistics_placeholder = st.empty() # Define statistics placeholder outside the loop
        while st.session_state.webcam_running:
            cap = st.session_state.webcam_cap
            ret, frame = cap.read()
            if ret:
                annotated_frame, results = predict_and_detect(model, frame.copy(), conf=confidence_threshold)
                webcam_placeholder.image(annotated_frame, channels="BGR", caption="Webcam Feed with Detections", width=webcam_width)
                
                # Update the content of the existing statistics placeholder
                with statistics_placeholder.container():
                    display_statistics(results)
            else:
                st.warning("Failed to grab frame from webcam. Stopping webcam.")
                st.session_state.webcam_running = False
                if 'webcam_cap' in st.session_state and st.session_state.webcam_cap is not None:
                    st.session_state.webcam_cap.release()
                    st.session_state.webcam_cap = None
                st.rerun()