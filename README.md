<div align="center">

# 🎭 Realtime Emotion Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Status](https://img.shields.io/badge/Project-Completed-success)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

Real-time Facial Emotion Recognition using Convolutional Neural Networks (CNN) and OpenCV.

</div>

---

## 📌 Overview

This project implements a **real-time emotion detection system** that:

- Detects faces using Haar Cascade classifier  
- Classifies facial expressions using a trained CNN model  
- Performs live emotion prediction via webcam  
- Displays emotion labels directly on video stream  

Supported emotions:

> 😠 Angry  
> 😃 Happy  
> 😢 Sad  
> 😐 Neutral  
> 😲 Surprise  
> 😨 Fear  
> 🤢 Disgust  

---

## 🧠 Model Architecture

The emotion classifier is built using a Convolutional Neural Network (CNN) consisting of:

- Convolutional layers  
- ReLU activation  
- MaxPooling layers  
- Flatten layer  
- Fully Connected layers  
- Softmax output layer  

Model file:

```
emotion_detection_model.h5
```

Input Shape: `48x48 Grayscale`

Loss Function: `Categorical Crossentropy`  
Optimizer: `Adam`

---

## 📊 Model Performance

| Metric | Value |
|--------|--------|
| Training Accuracy | ~94% |
| Validation Accuracy | ~88–90% |
| Input Size | 48x48 |
| Classes | 7 Emotions |

> Accuracy may vary depending on dataset split and training configuration.

---

## 📉 Confusion Matrix

The confusion matrix helps evaluate classification performance across emotions.

Example (Illustrative):

| Actual \ Predicted | Angry | Happy | Sad | Neutral | Surprise | Fear | Disgust |
|--------------------|-------|-------|-----|---------|----------|------|---------|
| Angry              | 92%   | 2%    | 3%  | 1%      | 1%       | 1%   | 0%      |
| Happy              | 1%    | 95%   | 1%  | 1%      | 2%       | 0%   | 0%      |
| Sad                | 4%    | 1%    | 90% | 3%      | 1%       | 1%   | 0%      |

This identifies misclassification trends and helps guide model improvements.

---

## 📂 Project Structure

```
Realtime-Emotion-Detection/
│
├── .gitignore
├── .gitattributes
├── README.md
├── requirements.txt
│
├── emotion_detection_model.h5
├── haarcascade_frontalface_default.xml
│
├── notebooks/
│   ├── CTTC_MODEL.ipynb
│   ├── CTTC_Project.ipynb
│   └── Detection.ipynb
│
├── data/
│   ├── images.p
│   └── labels.p
│
├── Emotion/
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   ├── neutral/
│   ├── surprise/
│   ├── fear/
│   └── disgust/
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/realtime-emotion-detection.git
cd realtime-emotion-detection
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 🔹 Real-Time Detection

Run:

```
notebooks/Detection.ipynb
```

OR convert to Python script:

```bash
python detection.py
```

Press **Q** to close webcam window.

---

## 🔬 How the System Works

1. Webcam captures frame  
2. Frame converted to grayscale  
3. Haar Cascade detects face region  
4. Face resized to 48x48  
5. Pixel values normalized  
6. CNN predicts emotion  
7. Label rendered on frame  

---

## 🧾 Important Repository Notes

- `.gitignore` excludes virtual environments, cache files, logs, and unnecessary system files.
- `requirements.txt` ensures reproducible environment setup.
- Haar Cascade XML files are included because they are required for face detection.
- Large datasets can be excluded if needed to keep repository lightweight.

---

## 📈 Future Enhancements

- Replace Haar Cascade with DNN-based face detector  
- Deploy using Flask / FastAPI  
- Add probability confidence bars  
- Convert into web-based interface  
- Deploy on edge devices  

---

## 🎯 Applications

- Human-Computer Interaction  
- Smart Surveillance  
- Emotion-aware AI systems  
- Classroom engagement tracking  
- Customer sentiment analysis  

---

## 🤝 Contributing

Pull requests are welcome.  
Open an issue for major feature discussions.

---

## 📜 License

For educational and research purposes.

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
