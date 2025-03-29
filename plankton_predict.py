import torch    
import cv2
import warnings
import os
import logging

from ultralytics import YOLO
from gradio_client import Client

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('RAILWAY_ENVIRONMENT') else 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def clean_text(text):
    return text

def qwen2(message):
    client = Client("Qwen/Qwen2-57b-a14b-instruct-demo")
    
    try:
        result = client.predict(
            query=message,
            history=[],
            system=
                f"""
                Saya ingin mendapatkan penjelasan mendalam tentang taksonomi dan klasifikasi plankton {message}. Berikut adalah beberapa aspek utama yang perlu dijelaskan:  
                1. *Hierarki Taksonomi dalam Plankton:*
                2. *Kelompok Utama Plankton Berdasarkan Taksonomi:*  
                3. *Klasifikasi Plankton Berdasarkan Kemampuan Bergerak:*  
                4. *Klasifikasi Berdasarkan Habitat:*  
                5. *Klasifikasi Berdasarkan Siklus Hidup:*  
                6. *Klasifikasi Berdasarkan Fungsi dalam Ekosistem:*  
                7. *Klasifikasi Berdasarkan Ukuran:*  
                """,
            api_name="/model_chat"
        )
    except:
        result = "Qwen tidak dapat memberikan penjelasan untuk query ini."
        
    return result[1][0][1]

def deepseek(message):
    client = Client("Abubekersiraj/Deepseek")

    try:
        result = client.predict(
            message=message,
            system_message=f"""
                Saya ingin mendapatkan penjelasan mendalam tentang taksonomi dan klasifikasi plankton {message}. Berikut adalah beberapa aspek utama yang perlu dijelaskan:  
                1. *Hierarki Taksonomi dalam Plankton:*
                2. *Kelompok Utama Plankton Berdasarkan Taksonomi:*  
                3. *Klasifikasi Plankton Berdasarkan Kemampuan Bergerak:*  
                4. *Klasifikasi Berdasarkan Habitat:*  
                5. *Klasifikasi Berdasarkan Siklus Hidup:*  
                6. *Klasifikasi Berdasarkan Fungsi dalam Ekosistem:*  
                7. *Klasifikasi Berdasarkan Ukuran:*  
                """,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.95,
            api_name="/chat"
        )
    except:
        result = "Deepseek tidak dapat memberikan penjelasan untuk query ini."

    return result

def predict_img(model_option, llm_option, img_path):
    if model_option == "yolov8-detect":
        model = YOLO("model/yolov8-detect.pt")
    elif model_option == "yolov8-acvit":
        model = YOLO("model/yolov8-acvit.pt")
    else:
        return "Model tidak ditemukan"
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (864, 576))

    results = model(img)

    # cv2.imwrite(
    #     filename="static/uploads/detect_img.jpg", 
    #     img=cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    # )
    cv2.imwrite(
        filename=f"{UPLOAD_FOLDER}/detect_img.jpg", 
        img=cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    )

    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    confidences = [float(box.conf) for box in results[0].boxes]

    if llm_option == "qwen":
        response = clean_text(qwen2(detected_classes))
    elif llm_option == "deepseek":
        response = clean_text(deepseek(detected_classes))
    else:
        response = "Pilih model LLM yang sesuai."

    return detected_classes, confidences, response