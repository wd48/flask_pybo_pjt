# pybo/rag_chat/rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

load_dotenv()
model_path = os.getenv("DATA_FILE_PATH")

# 1. 임베딩 모델
embedding_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    cache_folder=model_path
)

# 2025-08-06 문서 로딩 및 분할
def load_and_split_documents(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# 2025-08-06 Chroma 벡터 저장소 초기화
def initialize_chroma(docs, persist_dir="./chroma_db"):
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

# 2. Chroma 벡터 DB 로드 또는 생성
CHROMA_PATH = "./chroma_db"
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

vectordb = Chroma(
    collection_name="rag_kochat",
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH
)

# HuggingFace 관련 코드 제거
# Ollama를 사용하므로 HuggingFace 설정은 필요하지 않음

from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    {context}

    Question: {question}
    Answer:
    """
)

# Ollama 모델 사용
llm = Ollama(model="gemma3n:latest")

# RAG 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt}
)

# 질의 함수
def ask_rag(query: str):
    result = qa_chain.invoke(query)
    if isinstance(result, dict) and 'result' in result:
        answer_text = result['result']
        if 'Answer:' in answer_text:
            return answer_text.split('Answer:')[1].strip()
        return answer_text.strip()
    return result