import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification


# Fungsi untuk memuat model BERT dan tokenizer
PRE_TRAINED_MODEL = 'indobenchmark/indobert-base-p2'
bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
bert_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=2)


def predict_sentiment(text):
    input_ids = tf.constant(bert_tokenizer.encode(text, add_special_tokens=True))[None, :]  # Menambahkan token khusus [CLS] dan [SEP]
    logits = bert_model(input_ids)[0]
    probabilities = tf.nn.softmax(logits, axis=1)
    sentiment = tf.argmax(probabilities, axis=1)
    return sentiment.numpy()[0]


# Judul aplikasi
st.title('Prediksi Sentimen menggunakan BERT')

# Input teks
text = st.text_area('Masukkan teks', '')

# Tombol untuk memprediksi sentimen
if st.button('Prediksi'):
    if text.strip() == '':
        st.warning('Masukkan teks terlebih dahulu.')
    else:
        sentiment = predict_sentiment(text)
        if sentiment == 0:
            st.error('Sentimen: Negatif')
        else:
            st.success('Sentimen: Positif')
