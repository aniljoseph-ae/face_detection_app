import streamlit as st
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import pickle
import os

# Load MTCNN face detector
detector = MTCNN()

# Load Facenet model
facenet_model = load_model('models/facenet_model.h5')

# Load face embeddings and labels
EMBEDDINGS_FILE = 'models/Indian-celeb-embeddings.npz'
with np.load(EMBEDDINGS_FILE) as data:
    embeddings = data['arr_0']
    labels = data['arr_1']

# Function to extract face embeddings
def extract_face_embeddings(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    embedding = facenet_model.predict(face)
    return embedding

# Function to recognize faces
def recognize_faces(face_embedding):
    distances = np.linalg.norm(embeddings - face_embedding, axis=1)
    index = np.argmin(distances)
    return labels[index], distances[index]

# Function to add new face embeddings to the database
def add_new_face_embeddings(name, face_embedding):
    global embeddings, labels
    embeddings = np.append(embeddings, face_embedding, axis=0)
    labels = np.append(labels, [name], axis=0)
    np.savez_compressed(EMBEDDINGS_FILE, embeddings, labels)

# Admin password
ADMIN_PASSWORD = "admin123"

# Streamlit App Layout
st.title("Face Recognition System")

menu = ["Image", "Video", "Webcam", "Admin"]
choice = st.sidebar.selectbox("Select Input Mode", menu)

# --- IMAGE INPUT MODE ---
if choice == "Image":
    st.subheader("Image Face Recognition")
    image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Detect and recognize faces
        faces = detector.detect_faces(image)
        for face in faces:
            x, y, width, height = face['box']
            face_image = image[y:y+height, x:x+width]
            embedding = extract_face_embeddings(face_image)
            label, distance = recognize_faces(embedding)
            if distance < 1.0:  # Threshold for recognition
                cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
                cv2.putText(image, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2)
                cv2.putText(image, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Image")

# --- VIDEO INPUT MODE ---
elif choice == "Video":
    st.subheader("Video Face Recognition")
    video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

    if video_file is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and recognize faces
            faces = detector.detect_faces(frame)
            for face in faces:
                x, y, width, height = face['box']
                face_image = frame[y:y+height, x:x+width]
                embedding = extract_face_embeddings(face_image)
                label, distance = recognize_faces(embedding)
                if distance < 1.0:
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

# --- WEBCAM INPUT MODE ---
elif choice == "Webcam":
    st.subheader("Webcam Real-Time Face Recognition")
    run_webcam = st.button("Start Webcam")

    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and recognize faces
            faces = detector.detect_faces(frame)
            for face in faces:
                x, y, width, height = face['box']
                face_image = frame[y:y+height, x:x+width]
                embedding = extract_face_embeddings(face_image)
                label, distance = recognize_faces(embedding)
                if distance < 1.0:
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

# --- ADMIN MODE ---
elif choice == "Admin":
    st.subheader("Admin: Add New Faces to Database")
    password = st.text_input("Enter Admin Password", type="password")

    if password == ADMIN_PASSWORD:
        st.success("Access Granted. Capture Faces to Add to Database.")
        name = st.text_input("Enter Name for New Face Embeddings")

        if st.button("Start Webcam to Capture Faces"):
            cap = cv2.VideoCapture(0)
            face_data = []
            stframe = st.empty()

            while len(face_data) < 10:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect face
                faces = detector.detect_faces(frame)
                for face in faces:
                    x, y, width, height = face['box']
                    face_image = frame[y:y+height, x:x+width]
                    embedding = extract_face_embeddings(face_image)
                    face_data.append(embedding)
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()

            if len(face_data) > 0:
                # Add face embeddings to database
                for embedding in face_data:
                    add_new_face_embeddings(name, embedding)
                st.success(f"Face Embeddings for {name} Added Successfully!")
    else:
        st.error("Incorrect Password!")
