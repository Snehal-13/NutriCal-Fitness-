# importing libraries
import os
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import requests
from bs4 import BeautifulSoup


labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bellpeper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas',
          25: 'pineapple', 26: 'pomogranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweet corn', 32: 'sweet potato', 33: 'tomato', 34: 'turnip', 35: 'water melon'}


# loading model
model_path = 'C:\capstone cnn\FV.h5'
model = load_model(model_path)
model.make_predict_function()


app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def upload():
    imagefile = request.files['imagefile']
    imagepath = "./images/"+imagefile.filename
    imagefile.save(imagepath)
    result = model_predict(imagepath, model)
    cal = calories(result)
    sod = sodium(result)
    fat = fats(result)
    carb = carbs(result)
    pot = potasium(result)
    fib = fiber(result)
    # [cal,soduim,potasium,carbs,fats,fiber]
    return render_template("index.html", cal=str(cal), sod=str(sod), pot=str(pot), carb=str(carb), fat=str(fat), fib=str(fib), result=str(result).capitalize())


def calories(result):
    url = 'https://www.google.com/search?q=calories in '+result
    req = requests.get(url).text
    tmp = scrap = BeautifulSoup(req, 'html.parser')
    cal = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    return cal


def sodium(result):
    url = 'https://www.google.com/search?q=sodium in '+result
    req = requests.get(url).text
    tmp = scrap = BeautifulSoup(req, 'html.parser')
    soduim = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    return soduim


def fats(result):
    url = 'https://www.google.com/search?q=fats in '+result
    req = requests.get(url).text
    tmp = scrap = BeautifulSoup(req, 'html.parser')
    fats = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    return fats


def carbs(result):
    url = 'https://www.google.com/search?q=carbs in '+result
    req = requests.get(url).text
    tmp = scrap = BeautifulSoup(req, 'html.parser')
    carbs = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    return carbs


def potasium(result):
    url = 'https://www.google.com/search?q=potasium in '+result
    req = requests.get(url).text
    tmp = scrap = BeautifulSoup(req, 'html.parser')
    potasium = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    return potasium


def fiber(result):
    url = 'https://www.google.com/search?q=fiber in '+result
    req = requests.get(url).text
    tmp = scrap = BeautifulSoup(req, 'html.parser')
    fiber = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    return fiber


def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    ans = model.predict(img)
    y_class = ans.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res


if __name__ == '__main__':
    app.run(debug=True, port=3000)
