import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import streamlit as st
from PIL import Image

# Load MTCNN face detector
detector = MTCNN()

# Load Facenet model
facenet_model = load_model('models/facenet_model.h5')

# Load face embeddings and labels from npz file
with np.load('models/Indian-celeb-embeddings.npz') as data:
    embeddings = data['arr_0']
    labels = data['arr_1']

# Define function to extract face embeddings
def extract_face_embeddings(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    embedding = facenet_model.predict(face)
    return embedding

# Define function to recognize faces
def recognize_faces(face_embeddings):
    distances = np.linalg.norm(face_embeddings - embeddings, axis=1)
    indices = np.argmin(distances)
    return labels[indices]

# Streamlit app
st.title("Face Recognition with MTCNN and FaceNet")
st.write("Upload an image to detect and recognize faces.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = np.array(Image.open(uploaded_file))
    
    # Detect faces in the image
    faces = detector.detect_faces(image)

    if len(faces) == 0:
        st.write("No faces detected.")
    else:
        # Draw rectangles and labels on detected faces
        for face in faces:
            x, y, width, height = face['box']
            face_image = image[y:y+height, x:x+width]
            try:
                embedding = extract_face_embeddings(face_image)
                recognized_label = recognize_faces(embedding)
            except Exception as e:
                recognized_label = "Error"
            
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(image, recognized_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the output
        st.image(image, caption="Processed Image", use_column_width=True)
