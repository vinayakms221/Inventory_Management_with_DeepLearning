from __future__ import division, print_function
# coding=utf-8
import sys
import os
import cv2
import glob
from PIL import Image
import re
import numpy as np

# Keras

from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)


MODEL_PATH1 = './inventory0.h5'
MODEL_PATH2 = './inventory1.h5'
MODEL_PATH3 = './inventory2.h5'


model1 = load_model(MODEL_PATH1)
model2 = load_model(MODEL_PATH2)
model3 = load_model(MODEL_PATH3)

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(200, 200))

    # Preprocessing the image
    x = image.img_to_array(img)
 
    x=x/255
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    proba = model.predict(gray.reshape(1,200,200,1))
    # b=res[np.argmax(proba)]
    return proba


@app.route('/', methods=['GET'])
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
        d="/"
        # Make prediction
        preds = model_predict(file_path, model1)    
        res = ["bottom","top"]
        b=res[np.argmax(preds)]
        if(b=="top"):
            preds = model_predict(file_path, model2)    
            res = ["floral","plain"]
            c=res[np.argmax(preds)]
            fpath="categories/top"
            fpath=fpath+d+c
            ima = Image.open(file_path)
            file_path = os.path.join(basepath, fpath, secure_filename(f.filename))
            # f.save(file_path)
            ima.save(file_path, "JPEG")

        else:
            preds = model_predict(file_path, model3)    
            res = ["jeans","shorts"]
            c=res[np.argmax(preds)]
            fpath="categories/bottom"
            fpath=fpath+d+c
            ima = Image.open(file_path)
            file_path = os.path.join(basepath, fpath, secure_filename(f.filename))
            # f.save(file_path)
            ima.save(file_path, "JPEG")

        os.remove('./uploads/' + f.filename)
        
        b=b+d+c
        return b
    return None


if __name__ == '__main__':
    app.run(debug=True, threaded=False)

