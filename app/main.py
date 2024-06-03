import streamlit as st
from typing import Tuple
from PIL import Image
import cv2
from PIL.Image import Image as PImage
from datetime import datetime

MAX_FILE_SIZE = 200 * 1024 * 1024  # 5MB

@st.cache_resource
def load_models():
    from ultralytics import YOLO
    car_model = YOLO("./model/car.pt")
    transformer_model = YOLO("./model/transformer.pt")
    models = [car_model, transformer_model]
    return models

def rectangle_on_image(image, result, model) -> None:
    boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
    for box in boxes:  # Iterate over boxes
        r = box.xyxy[0].astype(int)  # Get corner points as int
        class_id = int(box.cls[0])  # Get class ID
        class_name = model.names[class_id]  # Get class name using the class ID
        print(f"Class: {class_name}, Box: {r}")  # Print class name and box coordinates
        cv2.rectangle(image, r[:2], r[2:], (0, 255, 0), 2)  # Draw boxes on the image


def model_annotate(model, image: PImage) -> Tuple[PImage,str]:
    result = model(source=image, save=True)
    # processed_image = cv2.imread('.png')
    detection_count = 0
    for res in result:
        detection_count += len(res.boxes.conf)
    
    if detection_count:
            return image,f"{detection_count} {result[0].names[int(result[0].boxes.cls.data[0])]} found"
    
    return image,"No detections"
        

def process_image(img: PImage) -> Tuple[PImage, str]:
    """ model recognize """
    models = load_models()

    for model in models:
        boxed_img, detections = model_annotate(model, img)

    if len(detections):
        return boxed_img, detections
    
    return img, "no detections"


def display_images(original_image, processed_image, detections: str) -> None:

    col1.empty()
    col2.empty()
    with col1:
        st.write("Original Image :camera:")
        st.image(original_image)

    with col2:
        st.write("Processed image :wrench:")
        st.image(processed_image)

    st.info("Result: " + detections)
    st.balloons()

    from io import BytesIO
    buf = BytesIO()
    processed_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.sidebar.download_button(
        label="Dowload processed image",
        data=byte_im,
        file_name="result"+str(datetime.now())+".png",
        mime="image/png"
    )   


st.set_page_config(layout="wide", page_title="Transformer recognizer")

st.write("## Transformer recognizer")
st.write(
"Upload image to see if there is a Transformer, car that has been Transformer's cover or just an ordinary car."
)

img_buff = st.camera_input("Take a photo")
col1, col2 = st.columns(2)

st.sidebar.write("## Upload")
my_upload = st.sidebar.file_uploader("Upload an image", type=["PNG"])

if img_buff is not None:
    image = Image.open(img_buff)
    processed_image, detections = process_image(image)
    display_images(image, processed_image, detections)

if my_upload is not None:
    
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 200MB.")
    else:
        image = Image.open(my_upload)
        processed_image, detections = process_image(image)
        display_images(image, processed_image, detections)

  
