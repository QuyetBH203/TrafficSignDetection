from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from yolov5 import detect  
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'yolov5/data/images'
DETECT_FOLDER = 'yolov5/runs/detect'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_input_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/detect/<path:filename>')
def serve_detected_image(filename):
    return send_from_directory('yolov5/runs/detect', filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        detect.run(weights='yolov5/models/best.pt', source=filepath, conf_thres=0.5)
        time.sleep(2)
        if not os.path.exists(DETECT_FOLDER):
            return "Error: Detection folder does not exist. Please check the detection process."

        exp_folders = sorted(os.listdir(DETECT_FOLDER), key=lambda x: os.path.getctime(os.path.join(DETECT_FOLDER, x)))
        if not exp_folders:
            return "Error: No 'exp' folder found in the detection directory."
        
        latest_exp_folder = exp_folders[-1]  
        output_image_filename = file.filename  
        output_image_path = os.path.join(DETECT_FOLDER, latest_exp_folder, output_image_filename)

        if not os.path.exists(output_image_path):
            return f"Error: Output file not found at {output_image_path}"

        return render_template('result.html', input_image=file.filename, output_image=f"{latest_exp_folder}/{output_image_filename}")

if __name__ == "__main__":
    app.run(debug=True)
