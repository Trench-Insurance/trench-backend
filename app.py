from flask import Flask, request, jsonify
import pickle
import joblib

app = Flask(__name__)

model = pickle.load(open("model.pickle", "rb"))
scaler = joblib.load("scaler.gz")


@app.route("/")
def home():
    return "Welcome to Trench Backend, type in /predict?month=xxx to predict"


@app.route("/predict")
def predict():
    n_month = request.args.get("month")
    prediction = model.predict(1185, 1185 + int(n_month) - 1)
    prediction = scaler.inverse_transform(prediction.reshape(-1,1))
    prediction = prediction.reshape(-1,)
    return jsonify(prediction.tolist())
