from flask import Flask, request, jsonify
import pickle
import joblib
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

model = pickle.load(open("model.pickle", "rb"))
scaler = joblib.load("scaler.gz")


@app.route("/")
@cross_origin()
def home():
    return "Welcome to Trench Backend, type in /predict?month=xxx to predict"


@app.route("/predict")
@cross_origin()
def predict():
    n_month = request.args.get("month")
    prediction = model.predict(1185, 1185 + int(n_month) - 1)
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
    prediction = prediction.reshape(-1,)
    return jsonify(prediction.tolist())
