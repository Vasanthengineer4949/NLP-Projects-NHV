import logging
from twilio.rest import Client
from fastapi import FastAPI, Request
from langchain_community.llms import CTransformers
from app import response
from urllib.parse import parse_qs
from dotenv import dotenv_values
config = dotenv_values(".env")

app = FastAPI()

TWILIO_ACCOUNT_SID=config["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN=config["TWILIO_AUTH_TOKEN"]
TWILIO_NUMBER=config["TWILIO_NUMBER"]

account_sid = TWILIO_ACCOUNT_SID
auth_token = TWILIO_AUTH_TOKEN
client = Client(account_sid, auth_token)
twilio_number = TWILIO_NUMBER 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm = CTransformers(model="openhermes-2-mistral-7b.Q8_0.gguf", model_type= "llama")

# Sending message logic through Twilio Messaging API
def send_message(to_number, body_text):
    try:
        message = client.messages.create(
            from_=f"whatsapp:{twilio_number}",
            body=body_text,
            to=f"whatsapp:{to_number}"
            )
        logger.info(f"Message sent to {to_number}: {message.body}")
    except Exception as e:
        logger.error(f"Error sending message to {to_number}: {e}")

@app.post("/")
async def reply(question:Request):
    llm_question = parse_qs(await question.body())[b'Body'][0].decode()
    try:
        chat_response = response(llm_question, llm)
        send_message("+91xxxxxxxxxx", chat_response)
    except:
         send_message("+91xxxxxxxxxx", "wait")
    return chat_response



