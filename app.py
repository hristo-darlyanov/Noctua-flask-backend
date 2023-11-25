from flask import Flask, jsonify, request, after_this_request
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

def average_pixels_in_block(masks, image_size):
    tresholds = 8
    treshold_range = image_size[0] / tresholds
    average_x_values = [image_size[1] / 2, image_size[1] / 2, image_size[1] / 2, image_size[1] / 2, image_size[1] / 2,  image_size[1] / 2]
    computed_points = []
        
    for mask in masks:
        mask = mask[np.argsort(mask[:, 1])]
        index = 0
        current_treshold_range = treshold_range
        
        for i in range(tresholds - 2):
            sum = 0
            left_points = 0
            total_points = 0
            while mask[index][1] <= current_treshold_range:
                sum += mask[index][0]
                total_points += 1
                if mask[index][0] <= image_size[1] / 2:
                    left_points += 1
                index+=1
                if index == mask.shape[0]:
                    index = 0
                    break
            
            if total_points != 0:             
                average_x_values[i] = (sum / int(total_points))
            
            current_treshold_range += treshold_range
            left_points = 0
        index = 0
         
        for i in range(0, tresholds - 2, 2):
            computed_points.append((average_x_values[i] + average_x_values[i+1]) / 2)
         
    return computed_points

def get_lr_weights(points, image_size):    
    image_size_x = image_size[1]
    weights = []
    for point in points:
        weights.append(point / image_size_x)
    
    return weights

def get_weight_for_frame(frame):
    result = model(frame)
    turning_weight = 0
    if len(result) != 0:
        for r in result:
            print(r.names)
            print(r.boxes.cls)
            print(r.orig_shape)
            
            # get the classes and setup empty mask array
            classes = r.boxes.cls.cpu().numpy()
            masks = []
            
            # extract the needed masks
            index = 0
            for mask in r.masks.xy:
                if classes[index] == 5:
                    masks.append(mask)
                index+=1    
            
            if len(masks) == 0:
                continue
            points = average_pixels_in_block(masks, r.orig_shape)
            weights = get_lr_weights(points, r.orig_shape)
            turning_weight = (weights[0] * 0.1 + weights[1] * 0.4 + weights[2] * 0.5) / 1
            
    return turning_weight

app = Flask(__name__)
model = YOLO('./best.pt')

@app.route('/json', methods=['POST'])
def send_json():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    img = Image.open(file)
    
    turning_weight = get_weight_for_frame(img)
    img.save('file.jpeg')

    return f'{turning_weight}'

if __name__ == '__main__':
    app.run()