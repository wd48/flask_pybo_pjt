# pybo/rag_chat/rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 환경변수 로드
load_dotenv()
model_path = os.getenv("DATA_FILE_PATH")

# 1. 임베딩 모델
embedding_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    cache_folder=model_path
)

# 2025-08-06 문서 로딩 및 분할
# 텍스트 파일 로딩 후 500자 단위로 나눈다
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

# 벡터DB가 존재하지 않으면 새로 생성
vectordb = Chroma(
    collection_name="rag_kochat",
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH
)

# HuggingFace 관련 코드 제거
# Ollama를 사용하므로 HuggingFace 설정은 필요하지 않음

from langchain.prompts import PromptTemplate

# 프롬프트 템플릿
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

# 2025-08-07 업로드 파일에 대한 질의 실행 함수 (LLM 호출)
def run_llm_chain(query, retriever):
    global llm  # 로컬에서 실행 중인 Ollama 모델
    #qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.invoke({"query": query})
    return result["result"]

# 2025-08-11 감정분석 체인 만들기
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

sentiment_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
        너는 텍스트의 감정을 분석하는 전문 감정 분석가야. 다음 문장을 읽고, 리뷰에 담긴 감정을 긍정(Positive), 부정(Negative), 중립(Neutral) 중 하나로 분류해줘. {text}
        감정을 분류한 뒤, 그 근거도 함께 설명해줘.
    """
)

sentiment_chain = LLMChain(
    llm=llm,
    prompt=sentiment_prompt
)

def analyze_sentiment(text: str):
    return sentiment_chain.run(text)