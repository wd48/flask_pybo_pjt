import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

UPLOAD_FOLDER = 'pybo/resources/uploads'
PERSIST_DIR = 'chroma_db'

embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

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
