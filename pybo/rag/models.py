# pybo/rag/models.py
import os
from flask import current_app
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# 전역 모델 변수
embedding_model = None
llm = None

# cuda/cpu 환경변수 적용,2025-09-02 jylee
load_dotenv()
gpu_mode = os.getenv("USE_CUDA")

# 임베딩 모델 호출, 2025-08-21 jylee
def get_embedding_model():
    """임베딩 모델을 로드하고 반환합니다. 모델이 이미 로드된 경우 기존 객체를 반환합니다."""
    global embedding_model
    if embedding_model is None:
        # 모델의 로컬 경로를 지정합니다.
        model_path = os.path.join(current_app.root_path, "..", "local_models", "jhgan_ko-sroberta-multitask")
        print(f"[-RAG-] Initializing embedding model from local path: {model_path}")

        # 모델이 GPU('cuda')를 사용하도록 명시적으로 지정합니다.
        model_kwargs = {'device': gpu_mode}

        embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs
        )
    return embedding_model

# 거대 언어 모델 호출 (LLM), 2025-08-21 jylee
def get_llm():
    """LLM을 로드하고 반환합니다. 모델이 이미 로드된 경우 기존 객체를 반환합니다."""
    global llm
    if llm is None:
        print(f"[-RAG-] Initializing LLM: {current_app.config['LLM_MODEL']}")
        llm = Ollama(
            base_url=current_app.config["LLM_HOST"],
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
