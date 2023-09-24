from langchain import LLMChain, PromptTemplate
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from config import *

class VasssGF:

    def __init__(self):

        self.gf_prompt = PromptTemplate(
            template=GF_PROMPT,
            input_variables=INP_VARS
        )
        self.llm = CTransformers(
                model = MODEL_CKPT,
                model_type = MODEL_TYPE,
                max_new_tokens = MAX_NEW_TOKENS,
                temperature = TEMPERATURE
            )
        self.memory = ConversationBufferMemory(k=K)

    def create_gf(self):
        gf = LLMChain(
            llm = self.llm,
            prompt=self.gf_prompt,
            memory=self.memory,
            verbose=True
        )
        return gf
    
