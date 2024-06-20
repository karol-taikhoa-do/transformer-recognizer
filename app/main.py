import streamlit as st
from PIL import Image
from PIL.Image import Image as PImage
from datetime import datetime

IMAGA_NAME='img.png'
IMAGE_DIR="images"
WORKING_IMAGE_DIR="runs/detect/predict"
FINAL_IMAGE_DIR="runs/detect/predict2"
CAR_MODEL_PATH="./model/car.pt"
TRANSFORMER_MODEL_PATH="./model/transformer.pt"

MAX_FILE_SIZE = 200 * 1024 * 1024  # 5MB


@st.cache_resource
def load_models():
    """
    laduje przy pierwszym uruchomieniu modele i zapisuje w cache
    programu dla pozniejszego szybszego dzialania.
    Usuwa poprzednie pliki lub przykladowe dla modeli (runs)
    dla poprawnego dzialania funkcji przetwarzania obrazu
    """
    from ultralytics import YOLO
    import shutil
    import os

    path = os.getcwd()
    shutil.rmtree(str(path) + "\\runs")

    car_model = YOLO(CAR_MODEL_PATH)
    transformer_model = YOLO(TRANSFORMER_MODEL_PATH)
    models = [car_model, transformer_model]
    return models


def model_annotate(model) -> str:
    """
    model przetwarza zdjecie z pliku i wynik detekcji nanosi na 
    osobne zdjecie (nietypowa wlasnosc funkcji 'predict', 
    nie widalem opcji custmizowanie WORKING_DIR).
    Wynikowe zdjecie zapisuje jako plik do dalszego odczytu,
    a funkcja zwraca wynik detekcji jako string 
    """
    result = model.predict(f"{IMAGE_DIR}/{IMAGA_NAME}", save=True,conf=0.5)
    annoted_image = Image.open(f"{WORKING_IMAGE_DIR}/{IMAGA_NAME}")
    detection_count = 0
  
    for res in result:
        detection_count += len(res.boxes.conf)
        res.show()

    annoted_image.save(f"{IMAGE_DIR}/{IMAGA_NAME}")
    
    if detection_count:
        res = f" {detection_count} {result[0].names[int(result[0].boxes.cls.data[0])]}"
        return res+"s" if detection_count > 1 else res
    
    return ""
        

def process_image() -> str:
    """
    uruchamia wszystkie modele i zapisuje skumulowane wyniki detekcji
    jako string. Jesli nic nie wykryly modele, zwroc odpowiednie info
    """

    detections = []

    for model in models:
        print(model.model_name)
        cur_detections = model_annotate(model)

        if len(cur_detections):
            detections.append(cur_detections)

    if len(detections):
        return ','.join(detections)+" found"
    
    return "No detections"


def display_images(original_image: PImage, processed_image: PImage, detections: str) -> None:
    """
    uzywa kolumn do wyswietlenia obrazow orginal, przetworzony przez modele i
    slowne podsumowanie skumulowanej deteckji przez modele.
    Ustawia na koniec przycisk na pobranie wynikowego zdjecia
    """
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

# MAIN
st.set_page_config(layout="wide", page_title="Transformer recognizer")
models = load_models()
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
    image.save(f"{IMAGE_DIR}/{IMAGA_NAME}")
    detections = process_image()
    processed_image = Image.open(f"{FINAL_IMAGE_DIR}/{IMAGA_NAME}")
    display_images(image, processed_image, detections)

if my_upload is not None:
    
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 200MB.")
    else:
        image = Image.open(my_upload)
        image.save(f"{IMAGE_DIR}/{IMAGA_NAME}")
        detections = process_image()
        processed_image = Image.open(f"{FINAL_IMAGE_DIR}/{IMAGA_NAME}")
        display_images(image, processed_image, detections)

  
