# pybo/rag/models.py
import os
import torch
from flask import current_app
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# 전역 모델 변수
embedding_model = None
llm = None

# .env 파일 로드
load_dotenv()

# 임베딩 모델 호출, 2025-08-21 jylee (CUDA 자동 감지 기능 추가, 2025-09-03 jylee)
def get_embedding_model():
    """임베딩 모델을 로드하고 반환합니다. 모델이 이미 로드된 경우 기존 객체를 반환합니다."""
    global embedding_model
    if embedding_model is None:
        # 모델의 로컬 경로를 지정합니다.
        model_path = os.path.join(current_app.root_path, "..", "local_models", "jhgan_ko-sroberta-multitask")
        print(f"[-RAG-] Initializing embedding model from local path: {model_path}")

        # CUDA 사용 가능 여부를 확인하고 장치를 동적으로 설정합니다.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[-RAG-] Embedding model will use device: {device}")
        model_kwargs = {'device': device}

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
