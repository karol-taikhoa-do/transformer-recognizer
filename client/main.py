import streamlit as st
from PIL import Image

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def process_image(img):
    """ model recognize """
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


# main screen
st.set_page_config(layout="wide", page_title="Transformer recognizer")

st.write("## Transformer recognizer")
st.write(
    "Upload image to see if there is a Transformer, car that has been Transformer's cover or just an ordinary car."
)

col1, col2 = st.columns(2)

# sidebar
st.sidebar.write("## Upload")
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
        
else:
    fix_image("./static/OptimusPrimeLK9562.jpg")


st.sidebar.write("Use camera") # todo
st.sidebar.button("Take a photo", type="primary")