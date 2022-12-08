from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import PIL.Image
# Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
# Model saved with Keras model.save()
MODEL_PATH = 'face_mask_detection_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(250, 250))

    # Preprocessing the image
    x = tf.keras.utils.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

@app.route("/", methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)         
        result = str(preds)# Convert to string
        if result=='[[0.]]':
            val='with mask'
        else:
            val='without mask'
        return val
    return None
if __name__ == '__main__':
    app.run(debug=True)
    
