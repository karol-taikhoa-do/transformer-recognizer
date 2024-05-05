import streamlit as st
from PIL import Image
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Transformer recognizer")

st.write("## Transformer recognizer")
st.write(
    "Upload image to see if there is a Transformer, car that has been Transformer's cover or just an ordinary car."
)
st.sidebar.write("## Upload")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
def process_image(img):
    """ model recognize """
    # from ultralytics import YOLO
    # model = YOLO("path_to_model")
    # results = model.predict(source=img)
    return img


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    col2.write("Processed image :wrench:")
    col2.image(process_image(image))
    st.sidebar.markdown("\n")
    st.info("Result: ")
    st.balloons()

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
photo_input = st.sidebar.button("Take a photo")

if photo_input:
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer:
        # To read image file buffer as a PIL Image:
        img = Image.open(img_file_buffer)
        fix_image(img)

        # To convert PIL Image to numpy array:
        img_array = np.array(img)

        # Check the type of img_array:
        # Should output: <class 'numpy.ndarray'>
        st.write(type(img_array))

        # Check the shape of img_array:
        # Should output shape: (height, width, channels)
        st.write(img_array.shape)

        data=process_image(img)
        st.sidebar.download_button(
            label="Dowload processed image",
            data=data,
            file_name="result"+str(datetime.now()),
        )
elif my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        data = process_image(my_upload)
        fix_image(upload=data)
        st.sidebar.download_button(
            label="Dowload processed image",
            data=data,
            file_name="result"+str(datetime.now())
        )        
else:
    fix_image("./static/OptimusPrimeLK9562.jpg")


