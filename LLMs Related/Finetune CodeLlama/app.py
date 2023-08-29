import gradio as gr
from ctransformers import AutoModelForCausalLM

title = "CodeLlama 7B GGUF Demo"

def load_model():
    model = AutoModelForCausalLM.from_pretrained("codellama-7b-instruct.Q4_K_M.gguf",
    model_type='llama',
    max_new_tokens = 64
    )
    return model

def chat_with_model(inp_chat, chat_history):

    model = load_model()
    response = model(
        inp_chat
    )
    return response

examples = [
    'Write the python code to train a linear regression model without using scikit-learn library.',
    'Write a Python code to generate even numbers from 0 to n given numbers',
    'Write a Python code to implement Stack data structure'
]

gr.ChatInterface(
    fn=chat_with_model,
    title=title,
    examples=examples
).launch()