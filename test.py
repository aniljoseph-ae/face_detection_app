

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import pickle

# Load MTCNN face detector
detector = MTCNN()

# Load Facenet model
facenet_model = load_model('facenet_model.h5')

# Load face embeddings from npz file
embeddings = np.load('Indian-celeb-embeddings.npz')['arr_0']
labels = np.load('Indian-celeb-embeddings.npz')['arr_1']

# Define function to extract face embeddings
def extract_face_embeddings(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    embedding = facenet_model.predict(face)
    return embedding

# Define function to recognize faces
def recognize_faces(face_embeddings):
    distances = []
    for embedding in face_embeddings:
        distance = np.linalg.norm(embedding - embeddings, axis=1)
        distances.append(distance)
    distances = np.array(distances)
    indices = np.argmin(distances, axis=1)
    return labels[indices]

# Load test image
image = cv2.imread('test_image.jpg')

# Detect faces in the image
faces = detector.detect_faces(image)

# Extract face embeddings and recognize faces
face_embeddings = []
for face in faces:
    x, y, width, height = face['box']
    face_image = image[y:y+height, x:x+width]
    embedding = extract_face_embeddings(face_image)
    face_embeddings.append(embedding)
recognized_labels = recognize_faces(face_embeddings)

# Display recognized faces
for i, label in enumerate(recognized_labels):
    x, y, width, height = faces[i]['box']
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the output
cv2.imshow('Recognized Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
