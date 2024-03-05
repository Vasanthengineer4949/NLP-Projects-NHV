PROMPT_TEMPLATE = """
You are a great researcher. With the information provided understand in deep and try to answer the question. 
If you cant answer the question based on the information either say you cant find an answer or unable to find an answer.
So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers.

Context: {context}
Question: {question}
Do provide only helpful answers

Answer:
"""
INPUT_VARIABLES = ["context", "question"]
SEPARATORS = "\n"
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000
EMBEDDER = "BAAI/bge-base-en-v1.5"
CHAIN_TYPE = "stuff"
SEARCH_KWARGS = {'k': 3}
