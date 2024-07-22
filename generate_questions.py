import os
import sys
import logging

from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever

from llama_index.vector_stores.lancedb import LanceDBVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# embeddings
from llama_index.embeddings.ollama import OllamaEmbedding

ollama_embedding = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

#from llama_index.llms.ollama import Ollama
#llm = Ollama(model="llama3", request_timeout=120.0)
#llm = Ollama(model="phi3:medium-128k", request_timeout=120.0)
#llm = Ollama(model="phi3:3.8b-mini-128k-instruct-q8_0", request_timeout=120.0)

"""
from llama_index.llms.together import TogetherLLM
llm = TogetherLLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key="b62edcbcd2e8ba84e80ab9261b2778d79faf11c89a0cf2e4c9832be03660e1a9",
)
"""

#from llama_index.llms.openai import OpenAI
#llm = OpenAI(model="gpt-4o-mini")
#llm = OpenAI(model="gpt-3.5-turbo")

from llama_index.llms.fireworks import Fireworks
llm = Fireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")

Settings.llm = llm
Settings.embed_model = ollama_embedding

DATA_DIR = "./data/"
documents = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True).load_data()

from llama_index.core.evaluation import DatasetGenerator

question_gen_query = (
    "You are customer support and technical expert of OpenShift aka OCP. Your task is to generate "
    "a set of frequently asked questions about OCP, its setup and debugging."
    "Formulate a single question that could be asked by a potential customer and user."
    "Restrict the question to the context information provided but don't mention the context or metadata itself."
    "And don't ask general questions not related to the service and the provided context."
)

dataset_generator = DatasetGenerator.from_documents(
    documents,
    question_gen_query=question_gen_query,
    num_questions_per_chunk=3
)

questions = dataset_generator.generate_questions_from_nodes(num=1000)
print(questions)
import pandas as pd
df = pd.DataFrame(questions, columns=['question'])
df.to_csv('ocp-q.csv', index=False)
