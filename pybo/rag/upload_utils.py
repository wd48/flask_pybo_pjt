import os
from flask import current_app
from typing import List, LiteralString
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pybo.rag.pipeline import (
    create_file_vectordb, get_file_vectordb,
    get_all_file_collections, generate_collection_name
)

# ChromaDB와 관련된 라이브러리 임포트
import chromadb
from config import CHAT_DB_PERSIST_DIR

# upload_folder is now defined within the functions that use it
# to avoid working outside of the application context.

# ChromaDB에 저장할 디렉토리 설정
# 1) save_pdf : 파일만 저장 (파일경로 반환)
def save_pdf(file_storage) -> LiteralString | str | bytes:
    upload_folder = current_app.config["CHAT_UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file_storage.filename)
    file_storage.save(filepath)
    print(f"[-RAG-] save_pdf() result: {filepath}")
    return filepath

# 저장된 pdf를 개별 컬렉션으로 인덱싱한다 (임베딩 및 벡터DB에 저장)
# 2) index_pdf : PDF 파일을 로드하고, 텍스트를 분할한 후 파일별 벡터DB에 저장합니다.
def index_pdf(filepath: str, chunk_size: int=500, chunk_overlap: int=50) -> int:
    filename = os.path.basename(filepath)
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    # PDF에서 텍스트를 추출하지 못한 경우 (예: 이미지로만 구성된 PDF)를 대비하여 필터링
    pages_with_content = [doc for doc in pages if doc.page_content and doc.page_content.strip()]
    
    if not pages_with_content:
        print(f"[-RAG-] Warning: No text could be extracted from {filename}. Skipping indexing.")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(pages_with_content)

    # 일부 문서가 비어있을 수 있으므로 최종적으로 다시 확인
    final_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
    if not final_docs:
        print(f"[-RAG-] Warning: Document splitting resulted in no content for {filename}. Skipping indexing.")
        return 0

    # 각 문서에 소스 정보 추가
    for doc in final_docs:
        doc.metadata["source"] = filepath
        doc.metadata["filename"] = filename

    # 파일별 개별 컬렉션 생성
    create_file_vectordb(filename, final_docs)

    print(f"[-RAG-] index_pdf() indexed {len(final_docs)} chunks from {filepath} into collection '{generate_collection_name(filename)}'")
    return len(final_docs)

# 3) 저장 + 인덱싱 헬퍼 : 추가된 청크 수 반환
def save_pdf_and_index(file_storage) -> int:
    filepath = save_pdf(file_storage)
    print(f"[-RAG-] save_pdf_and_index() saved file at: {filepath}")
    return index_pdf(filepath=filepath)

# 4) 업로드된 pdf 파일명 목록
def list_uploaded_pdfs() -> List[str]:
    upload_folder = current_app.config["CHAT_UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)
    print(f"[-RAG-] list_uploaded_pdfs() in folder: {upload_folder}")
    return sorted([f for f in os.listdir(upload_folder) if f.endswith('.pdf')])

# 5) 특정 pdf에만 한정된 retriever 생성 (개별 컬렉션에서)
def get_pdf_retriever(filename: str, k: int=3):
    """특정 파일의 개별 컬렉션에서 retriever를 생성합니다."""
    file_vectordb = get_file_vectordb(filename)
    if file_vectordb is None:
        print(f"[-RAG-] get_pdf_retriever() - No collection found for file: {filename}")
        return None

    print(f"[-RAG-] get_pdf_retriever() for file: {filename}, k={k}")
    return file_vectordb.as_retriever(search_kwargs={"k": k})

# chromaDB 컬렉션 이름 목록을 반환하는 함수
def get_collection_names() -> List[str]:
    """ChromaDB 컬렉션 이름 목록을 반환합니다."""
    try:
        persistent_client = chromadb.PersistentClient(CHAT_DB_PERSIST_DIR)
        collections = persistent_client.list_collections()
        collection_names = [c.name for c in collections]
        print(f"[-RAG-] Found {len(collection_names)} collections: {collection_names}")
        return collection_names
    except Exception as e:
        print(f"[-RAG-] Error getting collection names: {e}")
        return []

# 파일별 컬렉션 정보를 반환하는 함수
def get_file_collection_info() -> dict:
    """파일별 컬렉션 정보를 반환합니다."""
    persistent_client = chromadb.PersistentClient(current_app.config["CHAT_DB_PERSIST_DIR"])
    info = {}
    
    # 실제 업로드된 파일 목록을 기준으로 순회
    for filename in list_uploaded_pdfs():
        collection_name = generate_collection_name(filename)
        try:
            # ChromaDB에서 직접 컬렉션 정보를 가져옴
            collection = persistent_client.get_collection(name=collection_name)
            count = collection.count()
            info[filename] = {
                'collection_name': collection_name,
                'document_count': count
            }
        except ValueError:
            # 컬렉션이 존재하지 않는 경우
            info[filename] = {
                'collection_name': collection_name,
                'document_count': 0
            }
        except Exception as e:
            # 기타 예외 처리
            print(f"[-RAG-] Error getting collection info for {filename}: {e}")
            info[filename] = {
                'collection_name': collection_name,
                'document_count': 'Error'
            }
            
    return info

# 특정 파일의 컬렉션을 삭제하는 함수, 2025-08-20 jylee
def delete_collection_and_file(filename: str) -> bool:
    """특정 파일의 컬렉션과 물리적 파일을 모두 삭제합니다."""
    try:
        # 1. 물리적 파일 삭제
        upload_folder = current_app.config["CHAT_UPLOAD_FOLDER"]
        filepath = os.path.join(upload_folder, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"[-RAG-] Deleted physical file: {filepath}")
        else:
            print(f"[-RAG-] Physical file not found, skipping deletion: {filepath}")

        # 2. ChromaDB 컬렉션 삭제
        collection_name = generate_collection_name(filename)
        persistent_client = chromadb.PersistentClient(CHAT_DB_PERSIST_DIR)
        try:
            persistent_client.delete_collection(name=collection_name)
            print(f"[-RAG-] Deleted collection '{collection_name}'")
        except ValueError:
            print(f"[-RAG-] Collection '{collection_name}' not found, skipping deletion.")

        # 3. 메모리 캐시에서 제거
        if filename in get_all_file_collections():
            del get_all_file_collections()[filename]
            print(f"[-RAG-] Removed '{filename}' from in-memory cache.")
        
        return True
    except Exception as e:
        print(f"[-RAG-] Error during deletion for {filename}: {e}")
        return False
