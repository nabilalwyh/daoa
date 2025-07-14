import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model, tokenizer, dan label encoder
model = tf.keras.models.load_model("lstm_sentiment_model_70_30.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

max_len = 100  # Sesuaikan dengan panjang input saat training

# Judul dashboard
st.title("Analisis Sentimen Kalimat")
st.markdown("Masukkan kalimat berbahasa Indonesia untuk dianalisis sentimennya oleh model LSTM.")

# Input kalimat dari user
user_input = st.text_area("Tulis kalimat Anda di sini:")

# Tombol Prediksi
if st.button("Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("Kalimat tidak boleh kosong.")
    else:
        sequences = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        prediction = model.predict(padded)
        label_index = np.argmax(prediction)
        sentiment = label_encoder.inverse_transform([label_index])[0]
        
        st.subheader("Hasil Analisis:")
        st.write(f"**Sentimen:** {sentiment}")
        st.write(f"**Probabilitas:** {prediction[0][label_index]:.4f}")
