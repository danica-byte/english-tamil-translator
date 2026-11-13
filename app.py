
import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page configuration
st.set_page_config(
    page_title="English to Tamil Translator",
    page_icon="🌐",
    layout="centered"
)

# Load model only once
@st.cache_resource
def load_model():

    model_name = "Varsha00/finetuned-opusmt-en-to-ta"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.eval()

    return tokenizer, model


# Preprocess text
def preprocess_text(text):

    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    return text


# Translate text
def translate_text(text, tokenizer, model):

    if not text:
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', text)

    translated_sentences = []

    for sentence in sentences:

        if not sentence.strip():
            continue

        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():

            output = model.generate(
                **inputs,
                num_beams=4,
                max_length=128
            )

        translated = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        translated_sentences.append(translated)

    return " ".join(translated_sentences)


# Website heading
st.title("🌐 English to Tamil Translator")

st.write(
    "Translate English text into Tamil using "
    "a pretrained NLP machine translation model."
)

# Language selection
col1, col2 = st.columns(2)

with col1:
    st.selectbox("Source Language", ["English"])

with col2:
    st.selectbox("Target Language", ["Tamil"])

# User input
english_text = st.text_area(
    "Enter English text",
    placeholder="Type your English sentence here...",
    height=150
)

# Translation history
if "history" not in st.session_state:
    st.session_state.history = []

# Translate button
if st.button("Translate", type="primary"):

    clean_text = preprocess_text(english_text)

    if not clean_text:

        st.warning("Please enter some English text.")

    else:

        try:

            with st.spinner("Loading model and translating..."):

                tokenizer, model = load_model()

                tamil_text = translate_text(
                    clean_text,
                    tokenizer,
                    model
                )

            st.subheader("Tamil Translation")

            st.success(tamil_text)

            st.session_state.history.append(
                (clean_text, tamil_text)
            )

        except Exception as error:

            st.error("Translation failed. Please try again.")
            st.exception(error)


# Display translation history
st.divider()

st.subheader("Translation History")

if st.session_state.history:

    for number, (english, tamil) in enumerate(
        reversed(st.session_state.history), 1
    ):

        with st.expander(f"Translation {number}"):

            st.write("English:", english)
            st.write("Tamil:", tamil)

else:

    st.info("Your translation history will appear here.")
