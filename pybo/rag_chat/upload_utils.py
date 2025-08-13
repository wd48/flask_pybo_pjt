import os
from flask import current_app
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pybo.rag_chat.pipeline import embedding_model, vectordb, get_vectordb

# ChromaDB와 관련된 라이브러리 임포트
import chromadb
from config import CHAT_DB_PERSIST_DIR

upload_folder = current_app.config["CHAT_UPLOAD_FOLDER"]

# ChromaDB에 저장할 디렉토리 설정
# 1) save_pdf : 파일만 저장 (파일경로 반환)
def save_pdf(file_storage) -> str:
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file_storage.filename)
    file_storage.save(filepath)
    print(f"[-RAG-] save_pdf() result: {filepath}")
    return filepath

# 저장된 pdf를 인덱싱한다 (임베딩 및 벡터DB에 저장)
# 2) index_pdf : PDF 파일을 로드하고, 텍스트를 분할한 후 벡터DB에 저장합니다.
def index_pdf(filepath: str, chunk_size: int=500, chunk_overlap: int=50) -> int:
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(pages)
    # 각 문서에 소스 정보 추가
    pdf_vectordb = get_vectordb()

    pdf_vectordb.add_documents(docs)
    print(f"[-RAG-] index_pdf() indexed {len(docs)} chunks from {filepath}")
    return len(docs)

# 3) 저장 + 인덱싱 헬퍼 : 추가된 청크 수 반환
def save_pdf_and_index(file_storage) -> int:
    filepath = save_pdf(file_storage)
    print(f"[-RAG-] save_pdf_and_index() saved file at: {filepath}")
    return index_pdf(filepath=filepath)

# 4) 업로드된 pdf 파일명 목록
def list_uploaded_pdfs() -> List[str]:
    os.makedirs(upload_folder, exist_ok=True)
    print(f"[-RAG-] list_uploaded_pdfs() in folder: {upload_folder}")
    return sorted([f for f in os.listdir(upload_folder) if f.endswith('.pdf')])

# 5) 특정 pdf에만 한정된 retriever 생성
def get_pdf_retriever(filename:str, k: int=3):
    filepath = os.path.join(upload_folder, filename)
    retriever_vectordb = get_vectordb()
    print(f"[-RAG-] get_pdf_retriever() for file: {filepath}, k={k}")
    return retriever_vectordb.as_retriever(search_kwargs={"k": k, "filter": {"source": filepath}})

# 단일 PDF 파일을 쿼리하는 함수
def query_by_pdf(filename):
    filepath = os.path.join(upload_folder, filename)
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    local_vectordb = get_vectordb().from_documents(docs, embedding_model)
    print(f"[-RAG-] query_by_pdf() indexed {len(docs)} chunks from {filepath}")
    # 로컬 벡터 저장소를 검색 가능하게 설정
    return local_vectordb.as_retriever(search_kwargs={"k": 3})

# chromaDB 컬렉션 이름 목록을 반환하는 함수
def get_collection_names() -> List[str]:
    """ChromaDB 컬렉션 이름 목록을 반환합니다."""
    try:
        persistent_client = chromadb.PersistentClient(CHAT_DB_PERSIST_DIR)
        collections = persistent_client.list_collections()
        return [c.name for c in collections]
    except Exception as e:
        print(f"[-RAG-] Error getting collection names: {e}")
        return []
