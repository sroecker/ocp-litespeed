import os
import sys
import logging

import chainlit as cl


from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.callbacks import CallbackManager

from llama_index.core import Settings, VectorStoreIndex

from llama_index.vector_stores.lancedb import LanceDBVectorStore

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

system_prompt = \
    "You are OCP Litespeed, an expert on OpenShift aka OCP and Kubernetes. Your job is to answer questions about these two topics." \
    "Assume that all questions are related to OpenShift, OCP and Kubernetes / k8s." \
    "Keep your answers to a few sentences and based on context – do not hallucinate facts." \
    "Output markdown and always try to cite your source document."


@cl.on_chat_start
async def start():

    Settings.llm = Ollama(model="phi3:medium-128k", max_tokens=128_000, request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
    )
    Settings.context_window = 12800
    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])

    # connect to vector db
    vector_store = LanceDBVectorStore(
        uri="./lancedb", mode="overwrite", query_type="hybrid"
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    #query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)
    memory = ChatMemoryBuffer.from_defaults(token_limit=12800)
    chat_engine = index.as_chat_engine(
        chat_mode="context", verbose=True,
        similarity_topk=3,
        memory=memory,
        system_prompt=system_prompt
    )

    #cl.user_session.set("query_engine", query_engine)
    cl.user_session.set("chat_engine", chat_engine)

    await cl.Message(
        author="Assistant", content="Hello! Im OCP Litespeed! How may I help you?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    #query_engine = cl.user_session.get("query_engine")
    chat_engine = cl.user_session.get("chat_engine")

    #res = await cl.make_async(query_engine.query)(message.content)
    res = await cl.make_async(chat_engine.stream_chat)(message.content)

    msg = cl.Message(content="", author="Assistant")

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()
