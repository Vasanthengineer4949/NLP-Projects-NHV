DATA_DIR_PATH = "data/"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
EMBEDDER = "BAAI/bge-base-en-v1.5"
DEVICE = "cuda"
PROMPT_TEMPLATE = '''
With the information being provided try to answer the question. 
If you cant answer the question based on the information either say you cant find an answer or unable to find an answer.
So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

Context: {context}
Question: {question}
Dont make a sentence. Just provide the value. Do provide only helpful answers

Helpful answer:
'''
INP_VARS = ['context', 'question']
CHAIN_TYPE = "stuff"
SEARCH_KWARGS = {'k': 2}
MODEL_CKPT = "C:/Vasanth/Youtube Channel Prep/Langchain Projects/res_large/zephyr-7b-beta.Q8_0.gguf"
MODEL_TYPE = "mistral"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1