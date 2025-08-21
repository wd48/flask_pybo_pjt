# pybo/rag/models.py
from flask import current_app
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# 전역 모델 변수
embedding_model = None
llm = None

def get_embedding_model():
    """임베딩 모델을 로드하고 반환합니다. 모델이 이미 로드된 경우 기존 객체를 반환합니다."""
    global embedding_model
    if embedding_model is None:
        print(f"[-RAG-] Initializing embedding model: {current_app.config['EMBEDDING_MODEL']}")
        embedding_model = HuggingFaceEmbeddings(
            model_name=current_app.config["EMBEDDING_MODEL"]
        )
    return embedding_model

def get_llm():
    """LLM을 로드하고 반환합니다. 모델이 이미 로드된 경우 기존 객체를 반환합니다."""
    global llm
    if llm is None:
        print(f"[-RAG-] Initializing LLM: {current_app.config['LLM_MODEL']}")
        llm = Ollama(
            model=current_app.config["LLM_MODEL"],
            temperature=current_app.config["LLM_TEMPERATURE"]
        )
    return llm

def init_models():
    """애플리케이션 시작 시 모델을 미리 로드합니다."""
    print("[-RAG-] Pre-loading AI models...")
    get_embedding_model()
    get_llm()
    print("[-RAG-] AI models pre-loaded successfully.")
