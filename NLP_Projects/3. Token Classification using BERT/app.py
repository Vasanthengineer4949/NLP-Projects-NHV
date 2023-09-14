import streamlit as st
from transformers import pipeline

st.title("Keyword Extraction using BERT - NER")
st.subheader("This task aims at extract the key entities from the sentence")
@st.cache_resource(show_spinner=True)
def load_pipe(model_ckpt):
    pipe = pipeline("token-classification", model=model_ckpt, aggregation_strategy="simple")
    return pipe
token_classification_pipe = load_pipe("Vasanth/bert-ner-custom")

def inference(text_ipt):
    return token_classification_pipe(text_ipt)

text_input = st.text_input("Enter text")

if st.button("Extract Entities"):
    st.write(inference(text_input))
