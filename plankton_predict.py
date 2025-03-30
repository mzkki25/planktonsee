import torch    
import cv2
import warnings
import os
import logging
import firebase_admin
import tempfile
import os
import uuid
import google.generativeai as genai

from firebase_admin import credentials, firestore, storage
from ultralytics import YOLO
from gradio_client import Client

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_image_to_firebase(image_path):
    try:
        cred = credentials.Certificate("credential/all_credential.json")
        firebase_admin.initialize_app(cred) if not firebase_admin._apps else firebase_admin.get_app()
        db = firestore.client()
        bucket = storage.bucket(name='planktosee-temp-file')
        bucket.make_public()
        logger.info("Connected to Firestore successfully.")
    except Exception as e:
        logger.error(f"Error connecting to Firestore: {e}")

    blob = bucket.blob('images/' + os.path.basename(image_path))
    blob.upload_from_filename(image_path)
    blob.make_public()
    return blob.public_url

def gemini(message):
    try:
        genai.configure(api_key="")

        chatbot = genai.GenerativeModel("gemini-1.5-flash-002")
        system = f"""
            Saya ingin mendapatkan penjelasan mendalam tentang taksonomi dan klasifikasi plankton {message}. Berikut adalah beberapa aspek utama yang perlu dijelaskan:

            1. *Hierarki Taksonomi dalam Plankton:*  
            2. *Kelompok Utama Plankton Berdasarkan Taksonomi:*  
            3. *Klasifikasi Plankton Berdasarkan Kemampuan Bergerak:*  
            4. *Klasifikasi Berdasarkan Habitat:*  
            5. *Klasifikasi Berdasarkan Siklus Hidup:*  
            6. *Klasifikasi Berdasarkan Fungsi dalam Ekosistem:*  
            7. *Klasifikasi Berdasarkan Ukuran:*  
        """

        result = chatbot.generate_content(system).text

        file_path = os.path.join(UPLOAD_FOLDER, 'response.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(result)

        return "result result"

    except Exception as e:
        logger.error(f"Error di Gemini: {e}")
        return "Terjadi kesalahan saat menghubungi Gemini API."
    
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

        file_path = os.path.join(UPLOAD_FOLDER, 'response.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(result[1][0][1])
        
        return result[1][0][1]

    except Exception as e:
        logger.error(f"Error di Gemini: {e}")
        return "Terjadi kesalahan saat menghubungi Qwen2 API."
    
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

        file_path = os.path.join(UPLOAD_FOLDER, 'response.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        return result

    except Exception as e:
        logger.error(f"Error di Gemini: {e}")
        return "Terjadi kesalahan saat menghubungi DeepSeek"

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

    img_result = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_filename = uuid.uuid4().hex + ".jpg"
        cv2.imwrite(temp_filename, img_result)
    
    public_url = upload_image_to_firebase(temp_filename)

    os.remove(temp_filename)

    cv2.imwrite(
        filename=f"{UPLOAD_FOLDER}/detect_img.jpg", 
        img=cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    )

    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    confidences = [float(box.conf) for box in results[0].boxes]

    if llm_option == "qwen":
        qwen2(detected_classes)
    elif llm_option == "deepseek":
        deepseek(detected_classes)
    elif llm_option == "gemini":
        gemini(detected_classes)

    return detected_classes, confidences