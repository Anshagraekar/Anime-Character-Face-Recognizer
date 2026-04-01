# 🎌 Anime Character Face Recognition System

This project implements a deep learning–based anime character face recognition system using **EfficientNetB3 transfer learning**. The model accurately identifies anime characters from uploaded images and provides character trivia and streaming links through a Streamlit web application.

Anime faces differ from real human faces due to exaggerated features and artistic variation. This project addresses these challenges using a pretrained CNN fine-tuned on a curated anime dataset, achieving **92.42% test accuracy**.

---

## 🚀 Features
- 🎯 10-class anime character recognition
- 🧠 Transfer learning with EfficientNetB3
- 📈 92.42% test accuracy
- 🖼️ Image upload and real-time prediction
- 📚 Character trivia and anime streaming links
- 🌐 Streamlit web interface

---

## 🛠️ Tech Stack
- Python 3.12
- TensorFlow / Keras 2.19.1
- EfficientNetB3
- NumPy
- Pillow
- Streamlit
- Google Colab (GPU training)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | ~94.6% |
| Validation Accuracy | ~94.2% |
| Test Accuracy | **92.42%** |

---

## 📂 Project Structure
├── app.py
├── train.py
├── anime_character_classifier.h5 (external download)
├── anime_dataset_raw/ (not included in repo)
├── requirements.txt
├── .gitignore
└── README.md


---

## 📥 Model Download

Due to GitHub file size limits, the trained model is hosted externally:

🔗 **Download model here:**  
(Add your Google Drive / HuggingFace / Dropbox link)

After downloading, place it in the project root directory as:
anime_character_classifier.h5


---

## ▶️ Run Locally

``bash
pip install -r requirements.txt
streamlit run app.py
🎯 Use Cases
Anime character recognition

Image-based anime search

Fan content platforms

Computer vision demos

Educational deep learning projects

🔮 Future Improvements
Multi-face detection in group images

Mobile app deployment

Grad-CAM explainability

Support for more characters

Cloud deployment

🏆 Author
Ansh Agraekar
