import tensorflow as tf
import streamlit as st

max_len = 500

word_index = tf.keras.datasets.imdb.get_word_index()

model = tf.keras.models.load_model('simple_rnn_imdb.h5')

def preprocess_text(text):
    words = text.lower().split()
    encoded_text = [word_index.get(word,2)+3 for word in words]
    padded_text = tf.keras.preprocessing.sequence.pad_sequences([encoded_text],maxlen=max_len)
    return padded_text

st.title("IMDB Movie review sentiment Analysis")
st.write("Please enter your review to classify positive or negative")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    preprocessed_text = preprocess_text(user_input)

    prediction = model.predict(preprocessed_text)

    sentiment = 'Positive' if prediction[0][0] else 'Negative'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score:{prediction[0][0]}")
else:
    st.write("Please enter the movie review")