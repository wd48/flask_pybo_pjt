# pybo/rag/vectorstore.py
import os
import re
import hashlib

import chromadb
from flask import current_app
from langchain_chroma import Chroma

from . import models

# ChromaDB 클라이언트 인스턴스를 캐시하기 위한 전역 변수
persistent_client_instance = None

def get_persistent_client():
    """ChromaDB PersistentClient 인스턴스를 반환합니다. 이미 생성된 경우 캐시된 인스턴스를 반환합니다."""
    global persistent_client_instance
    if persistent_client_instance is None:
        try:
            persistent_client_instance = chromadb.PersistentClient(current_app.config["CHAT_DB_PERSIST_DIR"])
            print(f"[-RAG-] Initialized ChromaDB PersistentClient at {current_app.config["CHAT_DB_PERSIST_DIR"]}")
        except Exception as e:
            print(f"[-RAG-] Error initializing ChromaDB PersistentClient: {e}")
            # 오류 발생 시 None을 반환하거나 적절한 오류 처리를 할 수 있습니다.
            return None
    return persistent_client_instance

# 파일별 컬렉션을 저장하는 딕셔너리
file_collections = {}

# 파일명을 기반으로 컬렉션 이름을 생성하는 함수
def generate_collection_name(filename: str) -> str:
    """파일 이름으로부터 ChromaDB 컬렉션 이름을 생성합니다."""
    base_name = os.path.splitext(filename)[0].lower()

    # 1. 허용된 문자(a-z, 0-9, ., _, -)만 남기고 나머지는 밑줄로 대체
    cleaned_name = re.sub(r'[^a-z0-9._-]+', '_', base_name)

    # 2. 연속된 밑줄을 하나로 줄임
    cleaned_name = re.sub(r'_+', '_', cleaned_name)

    # 3. 이름의 시작과 끝이 [a-z0-9]가 되도록 처리
    #    밑줄이나 하이픈으로 시작하거나 끝나는 경우 제거
    cleaned_name = cleaned_name.strip('_- ') # Added space to strip

    # 4. 이름이 비어있으면 'default'로 설정
    if not cleaned_name:
        cleaned_name = "default"

    # 5. 최소 길이 3자 보장 (ChromaDB 요구사항)
    if len(cleaned_name) < 3:
        cleaned_name = "f" + cleaned_name # Prepend 'f' to ensure length and valid start

    # 6. 시작 문자가 [a-z0-9]가 아니면 'f' 추가
    if not cleaned_name[0].isalnum():
        cleaned_name = 'f' + cleaned_name

    # 7. 끝 문자가 [a-z0-9]가 아니면 'f' 추가
    if not cleaned_name[-1].isalnum():
        cleaned_name = cleaned_name + 'f'

    # 8. 파일명 해시를 추가하여 중복 방지
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    
    # 9. 최종 컬렉션 이름 생성 전, cleaned_name의 길이를 적절히 제한
    #    (예: 512 - len("file__") - len(file_hash) - 1 = 512 - 13 = 499자)
    max_cleaned_name_len = 499 
    if len(cleaned_name) > max_cleaned_name_len:
        cleaned_name = cleaned_name[:max_cleaned_name_len]

    final_collection_name = f"file_{cleaned_name}_{file_hash}"

    # 10. 최종 길이 제한 (ChromaDB max 512 chars)
    return final_collection_name[:512]

# 벡터 데이터베이스를 가져오는 함수 (모든 문서를 검색 대상으로 함)
def get_vectordb():
    """모든 문서의 내용을 담고 있는 전역 벡터DB를 반환합니다."""
    # 모든 파일 컬렉션을 가져와 하나의 ChromaDB 인스턴스로 통합
    all_collections = get_all_file_collections()
    if not all_collections:
        print("[-RAG-] No file collections available for global vector DB.")
        return None

    # 첫 번째 컬렉션을 기준으로 ChromaDB 인스턴스를 생성하고,
    # 이후 다른 컬렉션들을 추가하는 방식 (ChromaDB의 merge 기능이 없으므로)
    # 현재는 첫 번째 컬렉션을 반환하는 임시 방편 사용
    first_collection_key = next(iter(all_collections))
    return all_collections[first_collection_key]['vectordb']

# 벡터 데이터베이스 생성을 위한 내부 함수, 2025-08-19 jylee
def _create_vectordb_instance(docs=None, collection_name=None):
    """벡터 데이터베이스 인스턴스를 생성하는 내부 함수"""
    if docs is not None:
        # 문서로부터 새로운 vectordb 생성
        return Chroma.from_documents(
            documents=docs,
            embedding=models.get_embedding_model(),
            persist_directory=current_app.config["CHAT_DB_PERSIST_DIR"],
            collection_name=collection_name
        )
    else:
        # 기존 컬렉션 로드
        return Chroma(
            embedding_function=models.get_embedding_model(),
            persist_directory=current_app.config["CHAT_DB_PERSIST_DIR"],
            collection_name=collection_name
        )

# 파일별 벡터 데이터베이스를 생성하는 함수, 2025-08-19 jylee
def create_file_vectordb(filename: str, docs):
    """특정 파일에 대한 개별 벡터DB 컬렉션을 생성합니다."""
    collection_name = generate_collection_name(filename)

    # 내부 함수를 사용하여 vectordb 생성
    vectordb_instance = _create_vectordb_instance(docs=docs, collection_name=collection_name)

    # 파일 컬렉션 딕셔너리에 저장
    file_collections[filename] = {
        'vectordb': vectordb_instance,
        'collection_name': collection_name
    }

    print(f"[-RAG-] Created collection '{collection_name}' for file: {filename}")
    return vectordb_instance

# 파일별 벡터DB를 가져오는 함수
def get_file_vectordb(filename: str):
    """특정 파일의 벡터DB를 반환합니다."""
    if filename in file_collections:
        return file_collections[filename]['vectordb']

    # 기존 컬렉션이 있는지 확인
    collection_name = generate_collection_name(filename)
    try:
        # 내부 함수를 사용하여 기존 컬렉션 로드 (docs=None이므로 기존 컬렉션만 로드)
        vectordb_instance = _create_vectordb_instance(docs=None, collection_name=collection_name)

        file_collections[filename] = {
            'vectordb': vectordb_instance,
            'collection_name': collection_name
        }

        print(f"[-RAG-] Loaded existing collection '{collection_name}' for file: {filename}")
        return vectordb_instance
    except Exception as e:
        print(f"[-RAG-] Error loading collection for {filename}: {e}")
        return None

# 모든 파일 컬렉션 목록을 반환하는 함수
def get_all_file_collections():
    return file_collections