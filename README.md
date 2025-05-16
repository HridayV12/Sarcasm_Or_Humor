# 🧠 Sarcasm and Humor Classification from Memes (Multimodal ML Project)

This project implements a multimodal sarcasm and humor classification system that processes both the **text** and **image** components of memes using an ensemble of transformer-based NLP models and deep CNNs.

The final prediction is made using a **late fusion classifier** that combines features from:
- Text models: BERT, RoBERTa, DistilBERT, XLNet
- Image models: ResNet50, EfficientNet-B0, DenseNet121

The app is wrapped in a **Streamlit interface** allowing users to upload a meme image and receive sarcasm/humor predictions.

---

## 🗂 Project Structure


# Streamlit app interface
├── app2.py  
# Image and text feature extraction + prediction
├── fusion_model.py
# Training notebook for the late fusion classifier
├── Training.ipynb  
# Python package dependencies
├── requirements.txt
# Project documentation
├── README.md  
# Files and folders to exclude from Git
├── .gitignore




---

## 🚀 Setup Instructions

1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/sarcasm-humor-classifier.git
cd sarcasm-humor-classifier
```
2️⃣ Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
4️⃣ (Optional) Install CUDA for GPU Acceleration
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```



📦 Required Files (Manual Download)
Due to file size limits on GitHub, you must manually download the following:

images/	Meme image dataset Download https://drive.google.com/drive/folders/1S9J3AEpoT-Hm7vHxVgIapffn_oTJ9mB7?usp=sharing


🔍 Model Overview
📖 Text Feature Extractors
BERT (base, uncased)

RoBERTa (base)

DistilBERT

XLNet (base, cased)

🖼 Image Feature Extractors
ResNet50

EfficientNet-B0

DenseNet121

🧩 Late Fusion Classifier
Concatenates image and text features (7424D vector)

3-layer MLP with dropout

Trained on custom labeled meme dataset


🧠 Training Instructions
To retrain the model, open Training.ipynb and run all cells in order. You must:

Have images/ and labels.csv prepared

Use the same feature extraction code in fusion_model.py

Save the final model as late_fusion_model.pth

🧪 Run the Streamlit App
```bash
streamlit run app2.py
```
Video Link -> https://drive.google.com/file/d/12HNTThnzaShX5YJ3qtjzygAl_ntN1cS7/view?usp=sharing
