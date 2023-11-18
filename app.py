from flask import Flask, jsonify, request, after_this_request
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)
model = YOLO('./best.pt')
print(model.info())

@app.route('/json', methods=['POST'])
def send_json():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    img = Image.open(file)
    
    result = model(img)

    return 'File uploaded successfully'

if __name__ == '__main__':
    app.run()