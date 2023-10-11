from config import *
from codepal_retriever import create_codepal_retriever
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

def create_codepal():

    llm = CTransformers(
        model = MODEL,
        model_type = MODEL_TYPE,
        temperature = TEMPERATURE,
        max_new_tokens = MAX_NEW_TOKENS,
        do_sample = DO_SAMPLE
    )

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=INP_VARS)

    codepal = load_qa_chain(llm=llm, chain_type=CHAIN_TYPE, prompt=prompt)

    return codepal
    
