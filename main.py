import os
import logging

from flask import Flask, request, jsonify, render_template
from plankton_predict import predict_img

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.error("No file part in the request")
        return jsonify({
            "error": "No file part"
        }), 400
    file = request.files['file']
    
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({
            "error": "No selected file"
        }), 400
        
    if file:
        logger.info("File received: %s", file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        return jsonify({
            "img_path": img_path
        }), 200
    else:
        logger.error("File not allowed")
        return jsonify({
            "error": "File extension not allowed"
        }), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        data = request.json

        img_path = data.get('img_path', None)
        model_option = data.get('model_option', None)
        llm_option = data.get('llm_option', None)

        if not img_path or not os.path.exists(img_path):
            return jsonify({"error": "Image file not found"}), 400

        logger.info("Image path: %s", img_path)
        logger.info("Model option: %s", model_option)

        try:
            logger.info("Starting prediction")
            actual_class, probability_class = predict_img(model_option, llm_option, img_path)
            logger.info("Prediction completed")

            with open(f'{UPLOAD_FOLDER}/response.txt', 'r', encoding='utf-8') as f:
                response = f.read()
            
            return jsonify({
                "img_path": img_path,
                "actual_class": actual_class,
                "probability_class": probability_class,
                "response": response
            }), 200
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result')
def result():
    logger.info("Rendering result page")

    img_path = request.args.get('img_path')
    actual_class = request.args.getlist('actual_class')[0].split(",")
    probability_class = request.args.getlist('probability_class')[0].split(",")

    actual_classes = actual_class if len(actual_class) > 0 else ["Tidak terdeteksi"]
    probability_classes = probability_class if len(probability_class) > 0 else [99999]

    logger.info(f"Actual classes: {actual_classes}")
    logger.info(f"Probability classes: {probability_classes}")

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
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)