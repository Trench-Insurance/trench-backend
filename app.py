from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pickle", "rb"))


@app.route("/")
def home():
    return "Welcome to Trench Backend, type in /predict?month=xxx to predict"


@app.route("/predict")
def predict():
    n_month = request.args.get("month")
    prediction = model.predict(1185, 1185 + int(n_month) - 1)
    return jsonify(prediction.tolist())
