import streamlit as st
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.title("Cricket Match Commentary to Match Report Generation")
st.subheader("A prototype of an AI engine supported application that takes the match commentary as input and generates match report out of it")

@st.cache_resource(show_spinner=True)
def load_model_tokenizer():
    peft_model_id = "Vasanth/criccomm_to_cricnews"
    config = PeftConfig.from_pretrained(peft_model_id, from_transformers=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    model = PeftModel.from_pretrained(model, peft_model_id).to("cpu")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

def inference(model, tokenizer, input_sent):
    input_ids = tokenizer(input_sent, return_tensors="pt", truncation=True, max_length=1200).input_ids.to("cpu")
    outputs = model.generate(input_ids=input_ids, max_length=256, num_beams=5, no_repeat_ngram_size=3)
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

input_sent = st.text_area("Enter The commentary")
if st.button("Generate News"):
    st.subheader("Match Report")
    st.write(inference(model, tokenizer, input_sent))




