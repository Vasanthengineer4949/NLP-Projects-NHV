import streamlit as st
from vector_db import faiss_vector_db
from yt_buddy import YtBuddyCreator
from streamlit_chat import message

st.set_page_config(layout="wide")
st.title("YT Buddy - Chat with your YT videos")

def infer(bot, prompt):
    model_out = bot(prompt)
    answer = model_out['result']
    return answer

def display_conversation(history):
    for i in range(len(history["assistant"])):
        message(history["user"][i], is_user=True, key=str(i) + "_user")
        message(history["assistant"][i],key=str(i))

yt_url = st.text_input("Enter YT Video URL")
st.session_state.clicked = True

if "assistant" not in st.session_state:
        st.session_state["assistant"] = ["I am ready to help you"]
if "user" not in st.session_state:
        st.session_state["user"] = ["Hey there!"]

if yt_url and st.session_state.clicked==True:

    col1, col2 = st.columns([1,1])

    with col1:

        st.subheader("Video")
        st.video(yt_url)
    
    with col2:

        st.subheader("Chat with Video")

        with open("data/transcription.txt", "r+") as f:
            transcript = f.read()
            f.close()
            with st.expander("Transcript"):
                st.success(transcript)

            with st.expander("Chat with Video"):

                @st.cache_resource(show_spinner=True)
                def load_vector_store():
                    with st.spinner("Creation of Vector Store in Progress"):
                        vector_store = faiss_vector_db()
                        st.success(vector_store)
                load_vector_store()

                @st.cache_resource(show_spinner=True)
                def create_yt_buddy():
                        with st.spinner("Creation of YTBuddy in Progress"):
                            yt_buddy_creator = YtBuddyCreator()
                            yt_buddy = yt_buddy_creator.create_ytbuddy_bot()
                            st.success("YT_Buddy Created")
                            return yt_buddy
                yt_buddy = create_yt_buddy()

                user_query = st.text_input("Enter your question to the video")
                if st.button("Answer"):

                    answer = infer(yt_buddy, {'query': user_query})
                    st.session_state["user"].append(user_query)
                    st.session_state["assistant"].append(answer)

                    if st.session_state["assistant"]:
                        display_conversation(st.session_state)