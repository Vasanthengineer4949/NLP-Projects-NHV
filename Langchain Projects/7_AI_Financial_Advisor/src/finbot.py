from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from config import *
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
from ensemble_retriever import retriever_creation

class FinBotCreator:

    def __init__(self):
        self.prompt_temp = PROMPT_TEMPLATE
        self.input_variables = INP_VARS
        self.chain_type = CHAIN_TYPE
        self.search_kwargs = SEARCH_KWARGS
        self.embedder = EMBEDDER
        self.model_ckpt = MODEL_CKPT
        self.model_type = MODEL_TYPE
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE

    def create_custom_prompt(self):
        custom_prompt_temp = PromptTemplate(template=self.prompt_temp,
                            input_variables=self.input_variables)
        return custom_prompt_temp
    
    def load_llm(self):
        llm = CTransformers(
                model = self.model_ckpt,
                model_type=self.model_type,
                max_new_tokens = self.max_new_tokens,
                temperature = self.temperature
            )
        return llm
    
    def load_retriever(self):
        return retriever_creation()

    def create_bot(self, custom_prompt, retriever, llm):
        retrieval_qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type=self.chain_type,
                                retriever= retriever,
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": custom_prompt}
                            )
        return retrieval_qa_chain
    
    def create_finbot(self):
        self.custom_prompt = self.create_custom_prompt()
        self.retriever = self.load_retriever()
        self.llm = self.load_llm()
        self.bot = self.create_bot(self.custom_prompt, self.retriever, self.llm)
        return self.bot