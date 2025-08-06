import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

UPLOAD_FOLDER = 'pybo/resources/uploads'
PERSIST_DIR = 'chroma_db'

embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# ChromaDB에 저장할 디렉토리 설정
def save_and_embed_pdf(file_storage):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, file_storage.filename)
    file_storage.save(filepath)

    # PDF 로딩 및 텍스트 분할
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)

    # ChromaDB에 추가 저장
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    vectordb.add_documents(docs)
    vectordb.persist()

    return True

# 업로드된 PDF 파일 목록을 반환하는 함수
# 이 함수는 업로드된 PDF 파일의 목록을 반환합니다.
def list_uploaded_pdfs():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]

# 단일 PDF 파일을 임베딩하는 함수
# 이 함수는 지정된 PDF 파일을 로드하고, 텍스트를 분할한 후, ChromaDB에 임베딩합니다.
# 파일 이름을 인자로 받아 해당 파일을 임베딩합니다.
def embed_single_pdf(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    vectordb.add_documents(docs)
    vectordb.persist()
    return True

# 단일 PDF 파일을 쿼리하는 함수
def query_by_pdf(filename, query):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    vectordb = Chroma.from_documents(docs, embedding_model)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever