import logging
import sys
import os

from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.lancedb import LanceDBVectorStore
import textwrap


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#DATA_DIR = "./lightspeed-rag-content/ocp-product-docs-plaintext/4.15/"
DATA_DIR = "./data/"

documents = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True).load_data()
print("Document ID:", documents[0].doc_id, "Document Hash:", documents[0].hash)

# embeddings
from llama_index.embeddings.ollama import OllamaEmbedding

ollama_embedding = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# not really needed except for the service context
from llama_index.llms.ollama import Ollama
llm = Ollama(model="phi3:medium-128k", request_timeout=120.0)


Settings.llm = llm
Settings.embed_model = ollama_embedding
# TODO change parameters
Settings.context_window = 8096
Settings.node_parser = SentenceSplitter(chunk_size=4096, chunk_overlap=512)
Settings.num_output = 8096

vector_store = LanceDBVectorStore(
    uri="./lancedb", mode="overwrite", query_type="hybrid"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# FIXME embedding happens twice?
# or chunk size

index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    show_progress=True,
)
