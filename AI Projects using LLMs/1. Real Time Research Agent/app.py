import streamlit as st
from streamlit_chat import message
from researcher import Researcher
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
st.set_page_config(layout="wide")
st.session_state.clicked=True

@st.cache_resource(show_spinner=True)
def create_researcher():
    researcher = Researcher()
    return researcher
research_apprentice = create_researcher()

def display_conversation(history):
    for i in range(len(history["apprentice"])):
        message(history["user"][i], is_user=True, key=str(i) + "_user")
        message(history["apprentice"][i], key=str(i))

if st.session_state.clicked:
    st.title("RoboWiz - Your 24/7 AI Research Apprentice 🧑‍💻")
    st.subheader("An AI apprentice who can serve you 24/7 by researching on a given question in realtime and provide you answers accordingly")

    if "apprentice" not in st.session_state:
        st.session_state["apprentice"] = ["Hello. How can I help you?"]
    if "user" not in st.session_state:
        st.session_state["user"] = ["Hey RoboWiz!"]
    
    col1, col2 = st.columns([1,2])
    
    with col1:
        st.image("res/assistant.png")

    with col2:
        with st.expander("Command RoboWiz"):
            research_query_input = st.text_input("Resarch Query")
            if st.button("Send"):
                robowiz_output = research_apprentice.research(research_query_input)

                st.session_state["user"].append(research_query_input)
                st.session_state["apprentice"].append(robowiz_output)

                if st.session_state["apprentice"]:
                    display_conversation(st.session_state)