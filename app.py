import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import tensorflow as tf
from flask import Flask, render_template, request
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

app = Flask(__name__)

# ---- LOAD MODEL ONLY ONCE ----
model = TFDistilBertForSequenceClassification.from_pretrained("business_classifier_model")
tokenizer = DistilBertTokenizer.from_pretrained("business_classifier_model")

label_map = {0:"Technical",1:"Billing",2:"General"}

# ---- LOAD PRECOMPUTED METRICS ----
with open("metrics.json") as f:
    metrics = json.load(f)

def predict_text(text):

    enc = tokenizer(text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=64)

    output = model(**enc)
    pred = tf.argmax(output.logits, axis=1).numpy()[0]

    return label_map[pred]


@app.route("/", methods=["GET","POST"])
def home():

    result=""

    if request.method=="POST":
        result = predict_text(request.form["text"])

    return render_template("index.html",
            result=result,
            acc=metrics["accuracy"],
            f1=metrics["f1"],
            report=metrics["report"])


if __name__=="__main__":
    app.run(debug=True)
