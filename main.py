import os
import gunicorn

from flask import Flask, request, jsonify, render_template
from plankton_predict import predict_img

app = Flask(__name__)

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
        img_path = os.path.join('static/uploads', file.filename)
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
    data = request.json
    img_path = data['img_path']
    model_option = data['model_option']
    llm_option = data['llm_option']
    
    actual_class, probability_class, response = predict_img(model_option, llm_option, img_path)

    with open('static/uploads/response.txt', 'w', encoding='utf-8') as f:
        f.write(response)
    
    return jsonify({
        "img_path": img_path,
        "actual_class": actual_class,
        "probability_class": probability_class,
        "response": response
    }), 200

@app.route('/result')
def result():
    img_path = request.args.get('img_path')
    actual_class = request.args.getlist('actual_class')[0].split(",")
    probability_class = request.args.getlist('probability_class')[0].split(",")

    actual_classes = actual_class if len(actual_class) > 0 else ["Tidak terdeteksi"]
    probability_classes = probability_class if len(probability_class) > 0 else [99999]

    with open('static/uploads/response.txt', 'r', encoding='utf-8') as f:
        response = f.read()

    output_path = os.path.join('static/uploads', "detect_img.jpg")
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
    app.run(debug=False)