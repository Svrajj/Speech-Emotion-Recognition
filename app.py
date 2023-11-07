import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import sounddevice as sd
import streamlit as st
import warnings
from keras import models
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
warnings.filterwarnings('ignore')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
sample_rate = 44100
def record_audio(duration):
    with st.spinner('Recording...'):
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        return audio_data  # Return both audio data and sample rate

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60).T, axis=0)
    return mfcc
model = models.load_model('Models/best_model.h5') 


page = st.selectbox("Select a page", ["Home", "The Dataset", "Visualisation","Feature Extraction", "The Model", "Prediction", "Pretrained Model"])

if page == "Home":
    st.markdown(
        """

        # Speech Emotion Recognition using LSTM

        By Suraj Pradeepkumar

        [github](github.com/Svrajj), [linkedin](https://www.linkedin.com/in/svraj/)

        ---

        ## Applications
        * Customer Service
        * Mental Health
        * Human Computer Interaction
        * Market Research

        ---

        ## Approach

        * The Dataset
        * Visuals
        * Feature Extraction and preprocessing
        * The model
        * Prediction
        """
    )
   

elif page == "The Dataset":
    st.markdown(
        """
        # The Dataset

        ---

        ### We take the audio data from 4 major datasets
        
        * Crema (Crowd-sourced Emotional Multimodal Actors)
        * Ravdess (Ryerson Audio-Visual Database of Emotional Speech and Song)
        * Savee (Surrey Audio-Visual Expressed Emotion)
        * Tess (Toronto Emotional Speech Set)

        Dataset link: https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en

        ### We detect 7 different Emotions

        * Angry
        * Disgust
        * Fear
        * Happy
        * Neutral
        * Sad
        * Surprised

        ---
        ### Count of Audio files in ```.wav``` for each emotion

        """
    )

    df = {
        'Emotions':['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised'],
        'Count' : [1923, 1923, 1923, 1924, 1895, 1922, 652 ]
    }
    df = pd.DataFrame(df)
    st.dataframe(df)

    st.markdown(
        """
        ---
        ### Countplot
        """
    )
    st.image('Images/emotion countplot.png', caption='Emotions Countplot', use_column_width=True)
    df2 = {
        'paths':['/content/Capstone 2/Data/Sad/1086_TAI_SAD_XX.wav',
       '/content/Capstone 2/Data/Neutral/1070_TSI_NEU_XX.wav',
       '/content/Capstone 2/Data/Disgust/OAF_dead_disgust.wav',
       '/content/Capstone 2/Data/Sad/1022_IOM_SAD_XX.wav',
       '/content/Capstone 2/Data/Disgust/1081_TSI_DIS_XX.wav'],
       'labels':['Sad', 'Neutral', 'Disgust', 'Sad', 'Disgust']
    }
    df2 = pd.DataFrame(df2)
    st.markdown(
        """
        ---

        ### Creating a dataframe
        * We create a dataframe of the filepaths with their respective labels below
        * By running ```data.sample(5)``` we get the following output
        """
    )
    st.dataframe(df2)
    
   

elif page == "Visualisation":
    st.markdown(
        """
        # Visualisations

        ---

        ## Waveplot

        A waveplot is a visual representation of an audio waveform, showing how sound varies over time with amplitude on the y-axis and time on the x-axis.

        ### Function to plot the waveplot
        ```
        def waveplot(audio_path):
            # Load the audio file
            y, sr = librosa.load(audio_path)

            # Create a time array for the x-axis
            time = librosa.times_like(y)

            # Create a waveform plot
            plt.figure(figsize=(10, 4))
            plt.plot(time, y, color='b')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Audio Waveform')
            plt.show()
        ```

        ## Mel Spectrogram

        A mel spectrogram is a visual representation of audio that shows how different frequencies change over time. It's commonly used in signal processing and speech recognition to analyze audio data.

        ### Function to plot the Spectrogram
        ```
        def spectogram(data, sr, emotion):
            x = librosa.stft(data)
            xdb = librosa.amplitude_to_db(abs(x))
            plt.figure(figsize=(11,4))
            plt.title(emotion, size=20)
            librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
        ```


        ---

        ### Angry


        """
    )
    st.image('Images/Angry waveform.png', use_column_width=True)
    st.image('Images/Angry Spectogram.png', use_column_width=True)

    st.markdown(
        """
        ---

        ### Disgust
        """
    )
    st.image('Images/disgust waveform.png', use_column_width=True)
    st.image('Images/disgust Spectogram.png', use_column_width=True)
    st.markdown(
        """
        ---

        ### Fear
        """
    )
    st.image('Images/Fear waveform.png', use_column_width=True)
    st.image('Images/fear Spectogram.png', use_column_width=True)
    st.markdown(
        """
        ---

        ### Happy
        """
    )
    st.image('Images/Happy waveform.png', use_column_width=True)
    st.image('Images/Happy Spectogram.png', use_column_width=True)
    st.markdown(
        """
        ---

        ### Neutral
        """
    )
    st.image('Images/Neutral waveform.png', use_column_width=True)
    st.image('Images/Neutral Spectogram.png', use_column_width=True)
    st.markdown(
        """
        ---

        ### Sad
        """
    )
    st.image('Images/Sad waveform.png', use_column_width=True)
    st.image('Images/sad Spectogram.png', use_column_width=True)
    st.markdown(
        """
        ---

        ### Surprised
        """
    )
    st.image('Images/Surprised waveform.png', use_column_width=True)
    st.image('Images/Surprised Spectogram.png', use_column_width=True)

    


elif page == "Feature Extraction":
    st.markdown(
        """
        # Feature Extraction

        ---

        ## MFCC (Mel Frequency Cepstral Coefficients)
         MFCCs are like a set of numbers that describe the important parts of a sound, taking into account how our ears perceive it

         There are 2 parts to it:
         * Mel Scale: The Mel scale is a way to measure frequencies based on how our ears hear them
         * Cepstral analysis focuses on the rate of change in the sound.

         ---

         ## How to Calculate MFCCs using a spectrogram


        """
    )
    st.image('Images/Surprised Spectogram.png', use_column_width=True)
    st.markdown(
        """
        * Apply log to the Mel filterbanks
        * Apply DCT(Discrete Cosine Transform)
        * The resultants are our MFCCs

        ---

        ## Function to calculate MFCCs

        ```
        def extract_mfcc(filename):
            y, sr = librosa.load(filename, duration=3, offset=0.5)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60).T, axis=0)
            return mfcc
        ```
        * We would get 60 features because of ```n_mfcc=60```
        * We apply this function to each file
        * We then convert it into an array of shape ```(12162, 60)```
        * We then expand the dimensions to ```(12162, 60, 1)``` so that we can fit it in the deep learning model
        * We also apply OneHotEncoding to the labels , so the shape now becomes ```(12162, 7)``` 
        """
    )

elif page == "The Model":
    st.markdown(
        """
        # The Model

        ---

        This is how our model looks like:
        ```
        model = Sequential()

        model.add(layers.Conv1D(256, 5, input_shape=(60, 1), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.BatchNormalization())
        model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
        model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(7, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        ```

        ---

        ### Why use BiLSTM

        * Capture Temporal Patterns
        * Better Feature Extraction
        * Recuced Information Loss
        * Context Capture


        """
    )
elif page == "Prediction":
    st.markdown(
        """
        # Prediction

        ---
        """
    )
    try:
        audio_data = audio_recorder()
        if audio_data:
            st.audio(audio_data, format="audio/wav")

        # if st.button('Analyze Audio'):
        #         path2 = extract_mfcc(audio_data)  # Pass sample_rate to the function
        #         path2 = np.array(path2)
        #         path2 = np.expand_dims(path2, axis=0)
        #         path2 = np.expand_dims(path2, axis=2)
        #         pred2 = model.predict(path2, batch_size=64, verbose=1)
        #         pred2 = pred2.argmax(axis=1).item()
        #         pred2 = emotions[pred2]
        #         st.title(pred2)
        st.markdown(
            """ --- """
        )    
        path = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
        if path:
            st.audio(path, format='audio/wav')
        if st.button('Detect Emotion'):
            path = extract_mfcc(path)
            path = np.array(path)
            path = np.expand_dims(path, axis=0)
            path = np.expand_dims(path, axis=2)
            pred = model.predict(path, batch_size=64, verbose=1)
            emotions=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
            pred = pred.argmax(axis=1).item()
            pred = emotions[pred]
            st.title(pred)

        st.markdown(
            """
            ---
            """
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")

elif page == "Pretrained Model":
    st.markdown(
        """
        # Using Transfer Learning

        ---

        * We use ```harshit345/xlsr-wav2vec-speech-emotion-recognition``` for transfer learning
        * This uses ```Wav2Vec2``` model which was developed by ```Meta```
        * The model was trained on ```AESSD(Acted Emotional Speech Dynamic Database)```
        * It has an accuracy of ```80.6%``` on any audio

        This model can predict the following emotions

        * Anger
        * Disgust
        * Fear
        * Sadness
        * Happiness

        ---

        ### Try detecting from your own audio [here](https://colab.research.google.com/drive/1Nze8hb9aFjQk6BzemgXJOYAQiNRxmFnb?usp=sharing)

        """
    )
        



