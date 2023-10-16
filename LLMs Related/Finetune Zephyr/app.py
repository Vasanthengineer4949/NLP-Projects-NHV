from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
import streamlit as st
from streamlit_chat import message

st.session_state.clicked=True

def process_data_sample(example):

    processed_example = "<|system|>\n You are a support chatbot who helps with user queries chatbot who always responds in the style of a professional.</s>\n<|user|>\n" + example + "</s>\n<|assistant|>\n"

    return processed_example

@st.cache_resource(show_spinner=True)
def create_bot():

    tokenizer = AutoTokenizer.from_pretrained("Vasanth/zephyr-support-chatbot")

    model = AutoPeftModelForCausalLM.from_pretrained(
                                                        "Vasanth/zephyr-support-chatbot",
                                                        low_cpu_mem_usage=True,
                                                        return_dict=True,
                                                        torch_dtype=torch.float16,
                                                        device_map="cuda"
                                                    )

    generation_config = GenerationConfig(
                                            do_sample=True,
                                            temperature=0.5,
                                            max_new_tokens=256,
                                            pad_token_id=tokenizer.eos_token_id
                                        )
    
    return model, tokenizer, generation_config

model, tokenizer, generation_config = create_bot()

bot = create_bot()

def infer_bot(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, generation_config=generation_config)
    out_str = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, '')
    return out_str

def display_conversation(history):
    for i in range(len(history["assistant"])):
        message(history["user"][i], is_user=True, key=str(i) + "_user")
        message(history["assistant"][i],key=str(i))

def main():

    st.title("Support Member 📚🤖")
    st.subheader("A bot created using Zephyr which was finetuned to possess the capabilities to be a support member")

    user_input = st.text_input("Enter your query")

    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ["I am ready to help you"]
    if "user" not in st.session_state:
        st.session_state["user"] = ["Hey there!"]
                
    if st.session_state.clicked:
        if st.button("Answer"):

            answer = infer_bot(user_input)
            st.session_state["user"].append(user_input)
            st.session_state["assistant"].append(answer)

            if st.session_state["assistant"]:
                display_conversation(st.session_state)

if __name__ == "__main__":
    main()
    
    