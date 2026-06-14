import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(layout='wide')

with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipNet')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 

options = os.listdir(os.path.join('..', 'data', 's1'))
options = [f for f in options if f.endswith('.mpg')]  # Fixed: filter to .mpg only
selected_video = st.selectbox('Choose video', options)

col1, col2 = st.columns(2)

if options: 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i "{file_path}" -vcodec libx264 test_video.mp4 -y')
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        video_np = video.squeeze(-1)
        video_uint8 = ((video_np - video_np.min()) / (video_np.max() - video_np.min()) * 255).astype('uint8')
        imageio.mimsave('animation.gif', video_uint8, fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = st.cache_resource(load_model)()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)