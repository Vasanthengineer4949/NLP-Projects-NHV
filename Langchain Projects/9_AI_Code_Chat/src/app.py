import streamlit as st
from streamlit_chat import message
from codepal_retriever import create_codepal_retriever
from codepal import create_codepal
from config import *
st.set_page_config(layout="wide")
st.session_state.clicked=True

@st.cache_resource(show_spinner=True)
def create_codepal_pipeline():
    codepal_retriever = create_codepal_retriever()
    codepal_bot = create_codepal()
    return codepal_retriever, codepal_bot
codepal_retriever, codepal_bot = create_codepal_pipeline()

def display_conversation(history):
    for i in range(len(history["apprentice"])):
        message(history["user"][i], is_user=True, key=str(i) + "_user")
        message(history["apprentice"][i], key=str(i))

if st.session_state.clicked:
    st.title("Codepal - Your 24/7 Coding Partner 🧑‍💻")
    st.subheader("An AI coding partner who can help you 24/7 by making you understand a codebase on a given question in realtime and provide you answers accordingly")

    if "apprentice" not in st.session_state:
        st.session_state["apprentice"] = ["Hello. How can I help you?"]
    if "user" not in st.session_state:
        st.session_state["user"] = ["Hey CodePal!"]
    
    col1, col2 = st.columns([1,2])
    
    with col1:
        st.image("res/codepal.jpg")

    with col2:
        with st.expander("Ask CodePal"):
            code_query_input = st.text_input("Query")
            if st.button("Send"):
                docs = codepal_retriever.get_relevant_documents(code_query_input)
                print(docs)
                st.session_state["user"].append(code_query_input)
                st.session_state["apprentice"].append(codepal_bot(
                                                {
                                                    "input_documents": docs, 
                                                    "question": code_query_input
                                                },
                                                return_only_outputs=RETURN_ONLY_OUTPUTS
                                            )["output_text"])
                if st.session_state["apprentice"]:
                    display_conversation(st.session_state)