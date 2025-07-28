from flask import Blueprint, render_template, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from sympy.physics.units import temperature

bp = Blueprint('rag_chat', __name__, url_prefix='/rag')

# 임베딩 모델
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 벡터 스토어 (ChromaDB)
persist_directory = "chroma_db"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# LLM 모델 (Gemma 3n LLM)
llm = LlamaCpp(
    model_path=vectordb.model_path,
    n_ct=2048,
    temperature=0.1,
    max_tokens=512,
    verbose=False
)

