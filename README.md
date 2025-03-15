# Face Recognition with Age & Gender Detection

## Overview
This project is a deep learning-based **Face Recognition** system that detects faces, recognizes identities, and predicts age & gender from an image. The application is built using **TensorFlow, OpenCV, MTCNN, and FaceNet**, with a **Streamlit-based UI** for easy interaction.

## Features
- **Face Detection:** Uses **MTCNN** for detecting multiple faces in an image.
- **Face Recognition:** Trained a **FaceNet model** on an **Indian celebrity dataset**, storing embeddings for identity recognition.
- **Age & Gender Prediction:** Utilizes pre-trained **Caffe models** for accurate classification.
- **Interactive Web App:** Built with **Streamlit** for user-friendly image upload and real-time processing.

## Technologies Used
- **Deep Learning:** TensorFlow, Keras, FaceNet, MTCNN
- **Computer Vision:** OpenCV, NumPy
- **Pre-trained Models:** Caffe-based age & gender models
- **UI Framework:** Streamlit

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed. Then, clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/aniljoseph-ae/face_detection_app.git
cd face_detection_app

# Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Running the Streamlit App
To start the application, run the following command:

```bash
streamlit run app.py
```

### Upload an Image
1. Click on **"Choose an image..."** and upload a face image.
2. The app detects faces, recognizes identity, and predicts age & gender.
3. Processed image is displayed with labeled predictions.

## Folder Structure
```
face-recognition-app/
│── models/                # Pre-trained models & embeddings
│   ├── facenet_model.h5   # Trained FaceNet model
│   ├── Indian-celeb-embeddings.npz  # Stored embeddings & labels
│   ├── age_deploy.prototxt  # Age model prototxt
│   ├── age_net.caffemodel   # Age model weights
│   ├── gender_deploy.prototxt  # Gender model prototxt
│   ├── gender_net.caffemodel   # Gender model weights
│── app.py                 # Main Streamlit application
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

## Future Enhancements
- Extend face recognition dataset for improved generalization.
- Integrate real-time webcam support for live face recognition.
- Deploy as a web service using **FastAPI** or **Flask**.

## License
This project is open-source under the **MIT License**.

---

📌 *Developed by [Anil Joseph](https://github.com/aniljoseph-ae).* Feel free to contribute or raise issues! 🚀

