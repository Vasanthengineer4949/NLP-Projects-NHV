import streamlit as st
from streamlit_chat import message
from gf import VasssGF
from playsound import playsound
from dotenv import find_dotenv, load_dotenv
import os
import requests
import re
load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
st.set_page_config(layout="wide")
st.session_state.clicked=True

def replace_word_in_asterisks(input_string):
    pattern = r'\*([^*]+)\*'

    def replace(match):
        return "(" + match.group(1).upper() + ")"

    replacement_string = re.sub(pattern, replace, input_string)

    return replacement_string

@st.cache_resource(show_spinner=True)
def create_gf():
    vasssgf = VasssGF()
    return vasssgf.create_gf()
gf = create_gf()

def gf_conversation(gf, bf_input):
    gf_output = gf.predict(bf_input=bf_input)
    gf_output = replace_word_in_asterisks(gf_output)
    return gf_output

def display_conversation(history):
    for i in range(len(history["gf"])):
        message(history["bf"][i], is_user=True, key=str(i) + "_user")
        message(history["gf"][i], key=str(i))

def text_to_audio(message):
    data = {
            "text": message,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                                "stability": 0.0,
                                "similarity_boost": 0.0
                            }
            }
    
    headers = {
                'accept': 'audio/mpeg' ,
                'xi-api-key': ELEVEN_LABS_API_KEY,
                'Content-Type': 'application/json'
            }
    
    response = requests.post("https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?optimize_streaming_latency=0", json=data, headers=headers)
    if response.status_code == 200 and response.content:
        with open("gf_audio.mp3", "wb") as f:
            f.write(response.content)
        return response.content

if st.session_state.clicked:
    st.title("Vasss - Your AI Girlfriend 👩‍❤️‍👨")
    st.subheader("An AI girlfriend who can be a great companion who like to be naughty and flirting with you")

    if "gf" not in st.session_state:
        st.session_state["gf"] = ["Hello Babe..."]
    if "bf" not in st.session_state:
        st.session_state["bf"] = ["Hey Vasss!"]
    
    col1, col2 = st.columns([1,2])
    
    with col1:
        st.image("res/gf_img.jpg")

    with col2:
        with st.expander("Chat with Vasss"):
            bf_input = st.text_input("Message")
            if st.button("Send"):
                gf_message = gf_conversation(gf, bf_input)

                st.session_state["bf"].append(bf_input)
                st.session_state["gf"].append(gf_message)

                if st.session_state["gf"]:
                    display_conversation(st.session_state)
                    text_to_audio(gf_message)
                    playsound("gf_audio.mp3")

