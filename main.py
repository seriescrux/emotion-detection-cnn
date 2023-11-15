import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os
import pygame

# Initialize Streamlit session state
if 'detect_emotion_flag' not in st.session_state:
    st.session_state.detect_emotion_flag = False

st.title('Emotion Detection')

# Set up Pygame for music playback
pygame.mixer.init()

# Load the Haarcascades for face detection and the emotion detection model
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion music folders
emotion_music_folders = {
    'Angry': './songs/angry',
    'Happy': './songs/happy',
    'Sad': './songs/sad',
    'Disgust': './songs/disgust',
    'Fear': './songs/fear',
    'Neutral': './songs/neutral',
    'Surprise': './songs/surprise',
}

# Function to play music based on emotion
def play_music(emotion):
    folder_path = emotion_music_folders.get(emotion)
    if folder_path:
        songs = os.listdir(folder_path)
        if songs:
            # Select a random song
            song_path = os.path.join(folder_path, np.random.choice(songs))
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            print(f"Playing music for emotion: {emotion}")
        else:
            print(f"No songs found in the {emotion} folder.")
    else:
        print(f"No folder path defined for emotion: {emotion}")

# Camera feed
cap = cv2.VideoCapture(0)

# Emotion detection button
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    detect_emotion_button = st.button('Detect Emotion')

# Pause button
with row1_col2:
    pause_button = st.button('Pause Music')

# Display the video feed using st.image
video_placeholder = st.empty()

# Display detected emotion in a bigger font
detected_emotion_placeholder = st.empty()

while True:
    if detect_emotion_button:
        _, frame = cap.read()

        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                play_music(label)

                detected_emotion_placeholder.markdown(f'<h1 style="text-align: center;">{label}</h1>', unsafe_allow_html=True)

                # Set the state variable to True after emotion is detected
                detect_emotion_button = False

        # Display the video feed with detected emotion
        video_placeholder.image(frame, channels='BGR', use_column_width=True)

    else:
        # Display the video feed without emotion detection
        _, frame = cap.read()
        video_placeholder.image(frame, channels='BGR', use_column_width=True)

    if pause_button:
        pygame.mixer.music.pause()
    elif pygame.mixer.music.get_busy() == 0:
        play_music(emotion_labels[np.random.randint(7)])

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
