# Smart Text Classifier

A machine learning based text classification system that categorizes business queries into predefined categories:

- Technical  
- Billing  
- General  

The project uses **DistilBERT (Transformer model)** for classification and provides:

- Trained model  
- Web interface  
- Evaluation metrics  
- Confusion matrix visualization  

---
## Assumptions

- The input text is in English.
- User queries are short business-related messages.
- The dataset categories are limited to Technical, Billing, and General.
- The model is trained on representative business queries.
- The system runs on a machine with sufficient memory and CPU.
- The trained model is generated locally using main.py.
- Evaluation metrics are precomputed and loaded from metrics.json.
- The system is intended for academic and demo purposes.
---
  
## Features

- Text classification using Transformer (DistilBERT)  
- User-friendly web interface using Flask  
- REST API for integration  
- Displays:
  - Accuracy  
  - F1 Score  
  - Classification Report  
  - Confusion Matrix  
- Offline and completely free implementation  

---

## Technology Stack

- Python  
- TensorFlow  
- Transformers (HuggingFace)  
- Flask  
- Scikit-learn  
- Matplotlib & Seaborn  
- HTML, CSS, Bootstrap  


## Project Structure

business/

│
├── main.py  
│   → Model training, evaluation, saving metrics & confusion matrix  
│
├── app.py  
│   → Flask web application and prediction API  
│
├── metrics.json  
│   → Stored accuracy, F1-score and classification report  
│
├── business_classifier_model/  
│   → Saved DistilBERT trained weights and tokenizer  
│
├── static/  
│   └── cm.png  
│       → Confusion matrix visualization  
│
├── templates/  
│   └── index.html  
│       → User interface  
│
└── business_text_dataset.csv  
    → Business query dataset


## How to Run

### Train Model
python main.py

### Start Application
python app.py

Open → http://127.0.0.1:5000

---

## API

POST /api/predict

Request:
{ "text": "Internet not working" }

Response:
{ "category": "Technical" }
---

## Evaluation

- Accuracy  
- F1 Score  
- Classification Report  
- Confusion Matrix

---

## Use Cases

- Helpdesk routing  
- Email classification  
- Chatbot preprocessing  

---

## Future Work

- Add more categories  
- Probability display  
- User feedback  
- Database storage  

---

##  Developer

Afna V N  
Smart Text Classifier Project
