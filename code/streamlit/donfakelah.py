#import libraries
import streamlit as st
import librosa as lb
import numpy as np
import tensorflow as tf
from keras.models import load_model
import tensorflow.keras.backend as K
import base64

#define recall function
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


#define model function
def run_model():
    #recreate CNN and LSTM layers
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Reshape((16, 1024)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])
    #import weights
    model.load_weights(r'C:\Users\User\GA\sandbox\capstone_project\code\streamlit\model_weights.h5')
    
    #compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



#create function to process load and process audio file 
def audio_process(audio):
    desired_number_of_frames = 128
    number_of_mel_bins = 128    

    fixed_shape = (desired_number_of_frames, number_of_mel_bins)
    y, sr = lb.load(audio, sr=44100)
    mel_spec = lb.feature.melspectrogram(y=y, sr=sr)

    log_mel_spec = lb.power_to_db(mel_spec, ref=np.max)

    # Pad or truncate to ensure fixed shape
    if log_mel_spec.shape[1] < desired_number_of_frames:
        # Pad if the number of frames is less than desired
        pad_width = desired_number_of_frames - log_mel_spec.shape[1]
        padded_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80)  # Padding with -80 (for example)
        reshape = padded_spec.reshape(fixed_shape[0], fixed_shape[1], 1)
    elif log_mel_spec.shape[1] > desired_number_of_frames:
        # Truncate if the number of frames is more than desired
        truncate_spec = log_mel_spec[:, :desired_number_of_frames]
        reshape = truncate_spec.reshape(fixed_shape[0], fixed_shape[1], 1)
    else:
        # No need to pad or truncate if the shape matches the desired shape
        reshape = log_mel_spec.reshape(fixed_shape[0], fixed_shape[1], 1)

    # Convert lists to numpy arrays
    feature_array = reshape.reshape(1,128,128,1)
    
    #normalize the arrays
    min_val = np.min(feature_array)
    max_val = np.max(feature_array)
    normalized = (feature_array - min_val) / (max_val - min_val)
    
    

    return np.array(normalized)



st.set_page_config(page_title = 'DonFakeLah! - Deepfake Audio Detector')

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.75);  # Adjust white color overlay here and transparency (0.5)
        mix-blend-mode: lighten;
    }}
    </style>
    """
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg(r'C:\Users\User\GA\sandbox\capstone_project\code\streamlit\background.png')


st.title('Welcome to DonFakeLah!')
st.write("\n A platform where you can do a quick check on an audio authenticity")

#add line
st.markdown("""
    <hr style="border-top: 3px solid black;">
    """, unsafe_allow_html=True)

with st.chat_message('ai'):
    st.write('Hello! I can help you to identify if the audio is real or fake. Please upload the file into the chat :smile: ')
    

with st.chat_message('user'):
    audio = st.file_uploader(label = '', type =['wav','mp3'])



if audio is not None:
    st.audio(audio)
    if st.button("Check"):
        audio_data = audio_process(audio)
        model = run_model()
        y_pred = model.predict(audio_data)

        if np.argmax(y_pred) == 0:
            with st.chat_message('ai'):
                st.success("This audio is REAL!", icon="✅")
                
                
        else:
            with st.chat_message('ai'):
                st.error("Warning, it is FAKE!", icon="🚨")
                



