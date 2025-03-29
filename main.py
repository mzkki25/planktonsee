import os
import gunicorn

from flask import Flask, request, jsonify, render_template
from plankton_predict import predict_img

import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('RAILWAY_ENVIRONMENT') else 'static/uploads'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

            # actual_class = ["Plankton1", "Plankton2"]
            # probability_class = [0.95, 0.85]
            # response = "Plankton detected with high confidence."

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
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    # app.run(debug=False)
    port = int(os.environ.get("PORT", 8080)) 
    app.run(host="0.0.0.0", port=port, debug=True)