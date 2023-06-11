import streamlit as st
import pandas as pd
import joblib
import tensorflow_hub as hub
import tensorflow_text as text

st.write("""
# Mini AI Doctor

This app predicts your disease.
""")

@st.cache_resource
def encoder_txt():
  return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

embed = encoder_txt()
symptom = st.text_input('What Happened?',"", placeholder="I have no energy and have lost my appetite. I'm feeling really sick and don't know what's wrong.")
ask_embed = embed([symptom])

@st.cache_resource
def load_model():
  return joblib.load('LinearSVC1_model.joblib')

model = load_model()
disease = model.predict(ask_embed)
answer = disease[0]

st.write('Your symptom is', answer, '.')
