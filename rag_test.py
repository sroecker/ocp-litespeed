import os
import sys
import logging

from llama_index.core import Settings, VectorStoreIndex, StorageContext
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

from llama_index.llms.ollama import Ollama
#llm = Ollama(model="llama3", request_timeout=120.0)
llm = Ollama(model="phi3:medium-128k", request_timeout=120.0)


Settings.llm = llm
Settings.embed_model = ollama_embedding



vector_store = LanceDBVectorStore(
    uri="./lancedb", mode="overwrite", query_type="hybrid"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

queries = [
    "How to add a new worker node?",
    "How can I use NVIDIA GPUs?",
    "How to change a user password?"
]

"""
retriever = VectorIndexRetriever(
    index=index,
    similarity_topk=5,
)
response = retriever.retrieve(queries[0])
for node in response:
    print("node", node.score)
    print("node", node.metadata)
    #print("node", node.text)
    print("#####\n\n")
"""

from lancedb.rerankers import ColbertReranker
reranker = ColbertReranker()
vector_store._add_reranker(reranker)

query_engine = index.as_query_engine(
    #filters=query_filters,
    # vector_store_kwargs={
    #     "query_type": "fts",
    # },
)

response = query_engine.query(queries[2])
print(response)
print(response.metadata)
