
from llama_index.core import SimpleDirectoryReader, Document, StorageContext
DATA_DIR = "./data/"
#documents = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True).load_data()

filename_fn = lambda filename: {"file_name": filename}

documents = SimpleDirectoryReader(
    input_dir=DATA_DIR,
    recursive=True,
    num_files_limit=10,
    file_metadata=filename_fn,
).load_data()

#

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

from llama_index.llms.ollama import Ollama
generator_llm = Ollama(model="llama3", request_timeout=120.0)
#critic_llm = Ollama(model="llama3", request_timeout=120.0)
critic_llm = Ollama(model="tensortemplar/prometheus2:7b-fp16", request_timeout=120.0)

# embeddings
from llama_index.embeddings.ollama import OllamaEmbedding
embeddings = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

generator = TestsetGenerator.from_llama_index(
    generator_llm,
    critic_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_llamaindex_docs(
    documents,
    test_size=5,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)


df = testset.to_pandas()
df.to_csv("ollama_testset.csv", index=False)
