import streamlit as st
import pandas as pd
import joblib
import tensorflow_hub as hub
import tensorflow_text as text

st.write("""
# Mini AI Doctor

This app predicts your disease.
""")

symptom = st.text_input('What Happened?',"I have no energy and have lost my appetite. I'm feeling really sick and don't know what's wrong.")

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
ask_embed = embed([symptom])

model = joblib.load('LinearSVC1_model.joblib')
disease = model.predict(ask_embed)
answer = disease[0]

st.write('Your symptom is', answer, '.')