import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import seaborn as sns
import matplotlib.pyplot as plt

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


# 1. LOAD DATASET

df = pd.read_csv("business_text_dataset.csv")

label_map = {"Technical":0, "Billing":1, "General":2}
df["label"] = df["category"].map(label_map)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)


# 2. TOKENIZATION

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
test_enc  = tokenizer(test_texts,  truncation=True, padding=True, max_length=64)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_enc),
    train_labels
)).batch(8)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_enc),
    test_labels
)).batch(8)


# 3. LOAD MODEL

model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

optimizer = Adam(learning_rate=2e-5)

model.compile(
    optimizer=optimizer,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)


# 4. TRAIN MODEL

print("\nTraining Started...\n")

model.fit(
    train_dataset,
    epochs=3
)

model.save_pretrained("business_classifier_model")
tokenizer.save_pretrained("business_classifier_model")

print("\nModel Saved Successfully!\n")


# 5. EVALUATION

pred_probs = model.predict(test_dataset).logits
preds = np.argmax(pred_probs, axis=1)

acc = accuracy_score(test_labels, preds) * 100
f1  = f1_score(test_labels, preds, average="weighted") * 100

print("\n===== RESULTS =====")
print(f"Accuracy: {acc:.2f} %")
print(f"F1 Score: {f1:.2f} %\n")

print(classification_report(test_labels, preds,
      target_names=["Technical","Billing","General"]))


# ===== SAVE METRICS FOR DASHBOARD =====

import json

if not os.path.exists("static"):
    os.makedirs("static")

metrics = {
    "accuracy": float(acc),
    "f1": float(f1),
    "report": classification_report(
        test_labels, preds,
        target_names=["Technical","Billing","General"],
        output_dict=True
    )
}

with open("metrics.json","w") as f:
    json.dump(metrics,f)

print("Metrics saved!")


# ===== SAVE SMALL CONFUSION MATRIX =====

cm = confusion_matrix(test_labels, preds)

plt.figure(figsize=(3,3))       # SMALL SIZE
sns.set(font_scale=0.8)

sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Technical","Billing","General"],
            yticklabels=["Technical","Billing","General"],
            cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("static/cm.png")
plt.close()


# 6. PREDICTION FUNCTION

def predict_text(text):

    enc = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=64
    )

    output = model(**enc)

    pred = tf.argmax(output.logits, axis=1).numpy()[0]

    inv_map = {0:"Technical",1:"Billing",2:"General"}

    return inv_map[pred]


# 7. USER INPUT MODE

print("\n===== BUSINESS QUERY CLASSIFIER =====")
print("Type 'exit' to stop\n")

while True:

    user = input("Enter your query: ")

    if user.lower() == "exit":
        break

    print("\nPredicted Category:", predict_text(user), "\n")
