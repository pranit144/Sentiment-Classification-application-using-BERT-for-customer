import streamlit as st
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np

# Set layout to wide
st.set_page_config(layout="wide")


# Load the trained BERT model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    model = TFBertForSequenceClassification.from_pretrained('C:/Users/Pranit/PycharmProjects/customer/Model')
    tokenizer = BertTokenizer.from_pretrained('C:/Users/Pranit/PycharmProjects/customer/Tokenizer')
    return model, tokenizer


model, tokenizer = load_model()


# Tokenize and encode the input text
def encode_input(text, max_length=128):
    encoded_input = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',  # Updated for compatibility with TensorFlow
        return_attention_mask=True,
        return_tensors='tf'
    )
    return encoded_input['input_ids'], encoded_input['attention_mask']


# Prediction function
def predict_sentiment(text):
    input_ids, attention_mask = encode_input(text)
    prediction = model.predict([input_ids, attention_mask])[0]
    pred_label = np.argmax(prediction, axis=1)
    return pred_label[0], prediction[0]  # Return prediction scores


# Apply custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Background color */
    body {
        background-color: #f0f2f6;
    }
    /* Header font color */
    .stTitle {
        color: #3A3F44;
    }
    /* Text area color and font */
    .stTextArea {
        background-color: #ffffff;
        font-size: 18px;
    }
    /* Button color */
    div.stButton > button {
        background-color: #00A86B;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
    }
    /* Custom results style */
    .results {
        font-size: 20px;
        color: #007bff;
        font-weight: bold;
    }
    /* Icon style */
    .icon {
        vertical-align: middle;
        margin-right: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit App UI
st.title("Sentiment Classifier with BERT")

# Add icons from Font Awesome
st.write("""
    <div style='display: flex; align-items: center;'>
        <img src='https://img.icons8.com/ios-filled/50/000000/sentiment-analysis.png' class='icon' />
        <h3>Enter a sentence below and the model will predict whether it's Positive or Negative:</h3>
    </div>
""", unsafe_allow_html=True)

# User input
user_input = st.text_area("Enter Text:", "")

if st.button("🧠 Classify Sentiment"):
    if user_input:
        pred_label, prediction_scores = predict_sentiment(user_input)
        sentiment = "Positive" if pred_label == 1 else "Negative"

        # Display results
        st.markdown(f"<div class='results'>Predicted Sentiment: **{sentiment}**</div>", unsafe_allow_html=True)

        # Visualizations
        st.subheader("Text Analysis Results")
        st.write(f"**Word Count:** {len(user_input.split())}")
        st.write(f"**Character Count:** {len(user_input)}")

        # Display prediction scores
        st.write(f"**Positive Score:** {prediction_scores[1]:.2f}")
        st.write(f"**Negative Score:** {prediction_scores[0]:.2f}")

        # Visualize the sentiment scores
        st.bar_chart(prediction_scores)

    else:
        st.write("Please enter text to classify.")

st.write("---")
st.write("BERT Model fine-tuned for Sentiment Classification")
