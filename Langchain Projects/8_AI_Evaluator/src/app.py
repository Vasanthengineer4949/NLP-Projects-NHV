from eval import MistralEvaluator
import streamlit as st
st.set_page_config(layout="wide")
st.session_state.clicked=True
@st.cache_resource(show_spinner=True)
def load_evaluator():
    return MistralEvaluator()

st.title("Evaluating using Mistral7B 🕵️📓")
if __name__ == "__main__":
    mistral_evaluator = load_evaluator()
    input = st.text_area("Enter your input")
    prediction = st.text_area("Enter yout output")
    reference = st.text_area("Enter your reference")
    if st.button("Evaluate") and st.session_state.clicked==True:
        eval_sample = mistral_evaluator.eval_sample(
            input=input,
            prediction=prediction,
            reference=reference
        )
        for criteria, verdict in eval_sample.items():
            st.subheader(criteria.capitalize())
            st.write(verdict)