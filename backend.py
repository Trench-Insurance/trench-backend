from flask import Flask
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pickle","rb"))

@app.route("/predict")
def predict():
    n_month = request.args.get("month")
    prediction = model.predict(1185, 1185 + int(n_month) - 1)
    output = prediction[0]
    return jsonify(output)