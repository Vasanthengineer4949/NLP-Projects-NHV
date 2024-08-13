from langchain_ollama.llms import OllamaLLM
from langchain_groq.chat_models import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
from utils import math_exec, general_exec
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
os.environ["GROQ_API_KEY"] = os.getenv("GROQQ_API_KEY")

class Question_Type(BaseModel):
    question_type: str = Field(description="The type of question. One of two: math or general")

question_router_llm = ChatGroq(model="llama-3.1-8b-instant")
question_router = question_router_llm.with_structured_output(Question_Type)
math_model = OllamaLLM(model="math-llama3:latest")
general_model = ChatGroq(model="llama-3.1-8b-instant")


def chatbot(question_type, chat_history, question):
    if question_type == "math":
        print("Using math model...")
        return math_exec(math_model, question, chat_history)
    else:
        print("Using general model...")
        return general_exec(general_model, question, chat_history)

chat_history = [{"role": "assistant", "content": "I am an assistant created by Neural Hacks with Vasanth to help you out in general and math questions."}]
while True:
    question = input("Enter your question: ")
    question_router_formatted = """
    Identify the type of question whether it is math or general
    Question: {question}""".format(question=question)
    question_type = question_router.invoke(question_router_formatted).question_type
    chat_history = chatbot(question_type, chat_history, question)
    print(chat_history)
    print(chat_history[-1]["content"])

