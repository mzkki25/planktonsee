import os
import gunicorn
import time
import cv2

import torch    
import warnings

from ultralytics import YOLO
from gradio_client import Client

from flask import Flask, request, jsonify, render_template
# from plankton_predict import predict_img

import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)


# UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('RAILWAY_ENVIRONMENT') else 'static/uploads'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    # results = model(img)

    # logging.debug(f"Detected {len(results[0].boxes)} objects")
    # logging.debug(f"Detected classes: {results[0].boxes.cls}")

    cv2.imwrite(
        filename=f"{UPLOAD_FOLDER}/detect_img.jpg", 
        # img=cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )

    logging.debug(f"Image saved to {UPLOAD_FOLDER}/detect_img.jpg")

    # detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    # confidences = [float(box.conf) for box in results[0].boxes]

    # if llm_option == "qwen":
    #     response = qwen2(detected_classes)
    # elif llm_option == "deepseek":
    #     response = deepseek(detected_classes)
    # else:
    #     response = "Pilih model LLM yang sesuai."

    # logging.debug(f"Detected classes: {detected_classes}")

    detected_classes = ["Plankton_1", "Plankton_2"]
    confidences = [0.95, 0.85]
    response = "Ini adalah contoh respons dari model LLM."

    return detected_classes, confidences, response

@app.route('/')
def index():
    return render_template(template_name_or_list='opening.html')

@app.route('/action', methods=['GET'])
def action():
    return render_template(template_name_or_list='action.html')

@app.route('/opening')
def delete_upload():
    for file in os.listdir('static/uploads'):
        if file != 'original_image.jpg' and file != 'predicted_mask.jpg' and file != 'output_image.jpg':
            os.remove(os.path.join('static/uploads', file))
            
    return render_template(template_name_or_list='opening.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({
            "error": "No file part"
        }), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            "error": "No selected file"
        }), 400
        
    if file:
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)
        return jsonify({
            "img_path": img_path
        }), 200
    else:
        return jsonify({
            "error": "File extension not allowed"
        }), 400
    
@app.route('/predict', methods=['POST'])
def predict():
    logging.debug(f"Request received: {request.json}")

    try:
        data = request.json

        img_path = data.get('img_path', None)
        model_option = data.get('model_option', None)
        llm_option = data.get('llm_option', None)

        if not img_path or not os.path.exists(img_path):
            logging.error(f"Image path not found: {img_path}")
            return jsonify({"error": "Image file not found"}), 400

        logging.debug(f"Running prediction on {img_path}, model: {model_option}, llm: {llm_option}")
        
        try:
            actual_class, probability_class, response = predict_img(model_option, llm_option, img_path)

            logging.debug(f"Prediction result: {actual_class}, {probability_class}, {response}")
        except Exception as e:
            logging.error(f"Prediction error: {e}")

        with open(f'{UPLOAD_FOLDER}/response.txt', 'w', encoding='utf-8') as f:
            f.write(response)
        
        return jsonify({
            "img_path": img_path,
            "actual_class": actual_class,
            "probability_class": probability_class,
            "response": response
        }), 200
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/result')
def result():
    img_path = request.args.get('img_path')
    actual_class = request.args.getlist('actual_class')[0].split(",")
    probability_class = request.args.getlist('probability_class')[0].split(",")

    actual_classes = actual_class if len(actual_class) > 0 else ["Tidak terdeteksi"]
    probability_classes = probability_class if len(probability_class) > 0 else [99999]

    with open(f'{UPLOAD_FOLDER}/response.txt', 'r', encoding='utf-8') as f:
        response = f.read()

    output_path = os.path.join(f'{UPLOAD_FOLDER}', "detect_img.jpg")
    predictions = list(zip(
        [(" ".join(actual_class.split("_"))).title() for actual_class in actual_classes], 
        [f'{float(prob):.6f}' for prob in probability_classes]
    ))

    if img_path:
        return render_template(
            template_name_or_list='result.html', 
            img_path=output_path,
            predictions=predictions,
            response=response   
        )
    else:
        return render_template(template_name_or_list='opening.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080)) 
    app.run(host="0.0.0.0", port=port, debug=True)