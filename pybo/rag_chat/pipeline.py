# pybo/rag_chat/pipeline.py
import os
from dotenv import load_dotenv
from flask import current_app
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 환경변수 로드
load_dotenv()

#### 공통자원 초기화 ####
model_path = os.getenv("DATA_FILE_PATH")

# 전역 변수들을 None으로 초기화
embedding_model = None
vectordb = None
llm = None
qa_chain = None
sentiment_chain = None

# 임베딩 모델을 가져오는 함수
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print(f"[-RAG-] None get_embedding_model(), model_name: {current_app.config['EMBEDDING_MODEL']}")
        embedding_model = HuggingFaceEmbeddings(
            model_name=current_app.config["EMBEDDING_MODEL"]
        )
    print(f"[-RAG-] get_embedding_model() initialized with model: {current_app.config['EMBEDDING_MODEL']}")
    return embedding_model

# 벡터 데이터베이스를 가져오는 함수
def get_vectordb():
    global vectordb
    if vectordb is None:
        vectordb = Chroma(
            embedding_function=get_embedding_model(),
            persist_directory=current_app.config["CHAT_DB_PERSIST_DIR"]
        )
    print(f"[-RAG-] get_vectordb() initialized with collection: {vectordb._collection_name}")
    return vectordb

# LLM을 가져오는 함수
def get_llm():
    global llm
    if llm is None:
        llm = Ollama(
            model=current_app.config["LLM_MODEL"],
            temperature=current_app.config["LLM_TEMPERATURE"]
        )
    print(f"[-RAG-] get_llm() initialized with model: {current_app.config['LLM_MODEL']}")
    return llm

# RAG(검색 증강 생성) 체인을 가져오는 함수
def get_qa_chain(retriever):
    global qa_chain
    if qa_chain is None:
        # 프롬프트 템플릿
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
                {context}
                Question: {question}
                Answer:
            """
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": custom_prompt}
        )
    print(f"[-RAG-] get_qa_chain() initialized with LLM: {current_app.config['LLM_MODEL']}")
    return qa_chain

# 감정 분석 체인을 가져오는 함수
def get_sentiment_chain():
    global sentiment_chain
    if sentiment_chain is None:
        sentiment_prompt = PromptTemplate(
            input_variables=["gender", "age", "emotion", "meaning", "action", "reflect", "anchor"],
            template="""
                당신은 사용자의 감정 기록을 분석하는 전문 상담가입니다. 사용자의 다음 기록을 바탕으로 감정 상태를 진단하고, 행동이 어떤 의미를 가지는지 분석하여 정확하고 적절한 답변을 제공하세요.

                --- 감정 기록 ---
                성별: {gender}
                연령대: {age}
                걷기 전 감정: {emotion}
                감정을 느낀 이유: {meaning}
                도움이 된 행동: {action}
                행동 후 긍정적인 변화: {reflect}
                오늘의 한마디: {anchor}
                ---

                위 기록을 바탕으로 분석하고 답변해 주세요.
            """
        )

        sentiment_chain = LLMChain(
            llm=get_llm(),
            prompt=sentiment_prompt
        )
    print(f"[-RAG-] get_sentiment_chain() initialized with LLM: {current_app.config['LLM_MODEL']}")
    return sentiment_chain

# 2025-08-06 문서 로딩 및 분할
# 텍스트 파일 로딩 후 500자 단위로 나눈다
def load_and_split_documents(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    print(f"[-RAG-] load_and_split_documents() loaded {len(docs)} documents from {file_path}")
    return splitter.split_documents(docs)

# 2025-08-06 Chroma 벡터 저장소 초기화
def initialize_chroma(docs, persist_dir="./chroma_db"):
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print(f"[-RAG-] Chroma DB initialized with {len(docs)} documents.")
    return vectordb

# 전역 검색 함수 : 사용자가 입력한 질문에 대해 RAG(검색 증강 생성) 방식으로 답변을 생성하는 함수
def ask_rag(query: str):
    retriever = get_vectordb().as_retriever(search_kwargs={"k": 3})
    chain = get_qa_chain(retriever)
    result = chain.invoke(query)

    # 결과에서 'Answer:' 부분을 추출하여 반환
    if isinstance(result, dict) and 'result' in result:
        answer_text = result['result']
        print(f"[-RAG-] ask_rag() answer_text: {answer_text}")

        if 'Answer:' in answer_text:
            print(f"[-RAG-] ask_rag() found 'Answer:' in result")
            # return answer_text.split('Answer:')[1].strip()
        elif 'context:' in answer_text:
            print(f"[-RAG-] ask_rag() found 'context:' in result")
            lines = answer_text.split('\n')
            answer_lines = []
            skip_context = False
            for line in lines:
                if line.strip().startswith('context:'):
                    skip_context = True
                    continue
                if skip_context and line.strip() == '':
                    continue
                if not skip_context:
                    answer_lines.append(line)
            answer_text = '\n'.join(answer_lines)
            print(f"[-RAG-] ask_rag() cleaned answer_text: {answer_text}")
        return answer_text.strip()
    print(f"[-RAG-] ask_rag() unexpected result format: {result}")
    return result

# 2025-08-07 업로드 파일에 대한 질의 실행 함수 (LLM 호출)
# 특정 retriever를 사용하여 질문에 대한 답변을 생성하는 함수 : retriever는 PDF 파일에 대한 검색 기능을 제공
def run_llm_chain(query, retriever):
    chain = get_qa_chain(retriever)
    result = chain.invoke({"query": query})
    print(f"[-RAG-] run_llm_chain() result: {result}")
    return result["result"]

# 2025-08-11 감정 분석 함수 : 주어진 텍스트에 대해 감정 분석을 수행하는 함수
def analyze_sentiment(gender: str, age: str, emotion: str, meaning: str, action: str, reflect: str, anchor: str):
    chain = get_sentiment_chain()
    print(f"[-RAG-] analyze_sentiment() for emotional record")
    return chain.run(gender=gender,
        age=age,
        emotion=emotion,
        meaning=meaning,
        action=action,
        reflect=reflect,
        anchor=anchor)