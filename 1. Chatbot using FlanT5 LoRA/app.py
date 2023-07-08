import streamlit as st
from streamlit_chat import message
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.title("Chatbot")
@st.cache_resource(show_spinner=True)
def load_model_tokenizer():
    peft_model_id = "lora-flan-t5-large-chat"
    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    model = PeftModel.from_pretrained(model, peft_model_id).to("cpu")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

def inference(model, tokenizer, input_sent):
    input_ids = tokenizer(input_sent, return_tensors="pt", truncation=True, max_length=256).input_ids.to("cpu")
    outputs = model.generate(input_ids=input_ids, top_p=0.9, max_length=256)
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

message("Hi I am Flan T5 Chatbot. How can I help you?", is_user=False)

placeholder = st.empty()
input_ = st.text_input("Human")

if st.button("Generate"):
    with placeholder.container():
        message(input_, is_user=True)
    input_ = "Human: " + input_ + ". Assistant: "
    with st.spinner(text="Generating Response.....  "):
        with placeholder.container():
            message(inference(model, tokenizer, input_), is_user=False)



