# pybo/rag/pipeline.py
# ==============================================================================
# LangSmith Evaluation & Tracing Setup
# ==============================================================================
# For advanced evaluation and tracing of your chatbot's performance, it is
# highly recommended to use LangSmith.
#
# To enable LangSmith, you need to set the following environment variables.
# You can do this by creating a .env file in the root of your project and
# adding the lines below, or by setting them in your operating system.
#
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
# os.environ["LANGCHAIN_PROJECT"] = "YOUR_PROJECT_NAME" # e.g., "pybo-chatbot"
#
# Make sure you have the `langsmith` and `python-dotenv` packages installed:
# pip install langsmith python-dotenv
# ==============================================================================
import os
import time
from dotenv import load_dotenv
from flask import current_app
from operator import itemgetter
from langchain.chains import LLMChain, RetrievalQA, create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from . import models, vectorstore
from .metrics import log_chatbot_response_time

# 환경변수 로드
load_dotenv()

#### 공통자원 초기화 ####
model_path = os.getenv("DATA_FILE_PATH")

# 전역 변수들을 None으로 초기화
sentiment_chain = None

# RAG(검색 증강 생성) 체인을 가져오는 함수
def get_qa_chain(retriever):
    """RAG 체인을 생성합니다. retriever가 동적으로 변경되므로 체인을 캐시하지 않습니다."""
    # 프롬프트 템플릿
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            주어진 내용을 바탕으로 다음 질문에 대해 한국어로 답변해 주세요.
            ---
            {context}
            ---
            Question: {question}
            Answer:
        """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=models.get_llm(),
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    print(f"[-RAG-] QA chain created with LLM: {current_app.config['LLM_MODEL']}")
    return qa_chain

# 감정 분석 체인을 가져오는 함수 (RAG 기반 및 로깅 기능 추가, 2025-09-04 Gemini)
def get_sentiment_chain(config: dict):
    """
    RAG 기반의 감정 분석 체인을 생성하고 반환합니다.
    전문가 지식 베이스에서 관련 정보를 검색하여, 이를 바탕으로 LLM이 답변을 생성합니다.
    """
    # 1. 일관된 답변을 위해 전용 LLM 인스턴스 생성 (낮은 temperature)
    sentiment_llm = Ollama(
        base_url=config["LLM_HOST"],
        model=config["LLM_MODEL"],
        temperature=0.1
    )

    # 2. 지식 베이스를 검색하기 위한 Retriever 가져오기
    retriever = vectorstore.get_kb_retriever(k=2) # 관련성 높은 2개 문서 검색

    # 3. 새로운 RAG 프롬프트 템플릿
    sentiment_prompt = PromptTemplate(
        input_variables=["context", "gender", "age", "emotion", "meaning", "action", "reflect", "anchor"],
        template="""
            당신은 사용자의 감정 기록을 분석하는 전문 심리 상담가입니다.
            주어진 [전문가 조언]을 반드시 참고하여, 사용자의 기록을 바탕으로 아래 형식에 맞춰 답변을 생성해 주세요.

            --- 전문가 조언 ---
            {context}
            --- END OF ADVICE ---

            --- 감정 기록 ---
            - 성별: {gender}
            - 연령대: {age}
            - 걷기 전 감정: {emotion}
            - 감정을 느낀 이유: {meaning}
            - 도움이 된 행동: {action}
            - 행동 후 긍정적인 변화: {reflect}
            - 오늘의 한마디: {anchor}
            ---

            --- 분석 답변 형식 ---
            1.  **감정 진단**: [사용자의 감정 상태에 대한 진단]
            2.  **행동 분석**: [기록된 행동의 의미와 효과에 대한 분석]
            3.  **전문가 제언**: [전문가 조언(context)을 바탕으로 한 상담가로서의 조언이나 격려]
            ---

            위 '분석 답변 형식'에 맞춰서만 답변을 작성해 주세요.
        """
    )

    # 4. RAG 체인 구성 (LCEL 방식) 및 로깅 추가
    def log_and_format_docs(docs):
        print("\n--- [RAG Sentiment Analysis] Retrieved Context: ---")
        if not docs:
            print("No relevant context found in Knowledge Base.")
            return "참고할 전문가 조언을 찾지 못했습니다."
        
        for doc in docs:
            source = doc.metadata.get('filename', 'N/A')
            content_preview = doc.page_content.replace('\n', ' ')[:100]
            print(f"  - SOURCE: {source}")
            print(f"    CONTENT: {content_preview}...\n")
        print("-----------------------------------------------------\
")
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": itemgetter("query") | retriever | log_and_format_docs,
            "gender": itemgetter("gender"),
            "age": itemgetter("age"),
            "emotion": itemgetter("emotion"),
            "meaning": itemgetter("meaning"),
            "action": itemgetter("action"),
            "reflect": itemgetter("reflect"),
            "anchor": itemgetter("anchor")
        }
        | sentiment_prompt
        | sentiment_llm
        | StrOutputParser()
    )

    print(f"[-RAG-] Initialized RAG-based sentiment chain with LLM: {config['LLM_MODEL']}")
    return rag_chain



# 대화형 RAG 체인 생성 함수, 2025-08-27 jylee
def get_conversational_rag_chain(retriever):
    """
    대화 기록을 바탕으로 질문을 재작성하고, 문서를 검색하여 답변을 생성하는 대화형 RAG 체인을 생성합니다.
    """
    llm = models.get_llm()

    # 1. 질문 재작성(Query Rewriting)을 위한 프롬프트
    contextualize_q_system_prompt = (
        "주어진 채팅 기록과 사용자의 최근 질문을 바탕으로, "
        "채팅 기록을 참조할 필요가 없는 독립적인 질문으로 바꾸어 주세요. "
        "답변은 하지 말고, 필요한 경우 질문만 다시 만들어 주세요."
    )
    # 1-a. 질문 재작성 프롬프트 템플릿
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # 1-b. 질문 재작성 체인 생성
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. 최종 답변 생성을 위한 프롬프트
    qa_system_prompt = (
        "당신은 주어진 컨텍스트(context)에서만 질문에 답변하는 AI 어시스턴트입니다. "
        "정확하고 간결하게, 한국어로 답변해 주세요.\n\n"
        "{context}"
    )
    # 2-a. 최종 답변 프롬프트 템플릿
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # 재작성된 질문과 검색된 문서를 받아 답변을 생성하는 체인 생성
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. 위 두 체인을 하나로 결합
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# 전역 검색 함수 : 사용자가 입력한 질문에 대해 RAG(검색 증강 생성) 방식으로 답변을 생성하는 함수
def ask_rag(query: str):
    # 모든 파일 컬렉션에서 리트리버를 가져와 통합 검색을 수행합니다.
    all_collections = vectorstore.get_all_file_collections()
    if not all_collections:
        print("[-RAG-] No file collections available for global RAG search.")
        return "현재 검색할 수 있는 문서가 없습니다."

    # 임시 방편: 첫 번째 컬렉션의 리트리버를 사용합니다.
    # 실제 전역 검색을 위해서는 모든 컬렉션의 리트리버를 통합하는 로직이 필요합니다.
    first_collection_key = next(iter(all_collections))
    retriever = all_collections[first_collection_key]['vectordb'].as_retriever(search_kwargs={"k": 3})
    
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
    start_time = time.time()
    result = chain.invoke({"query": query})
    end_time = time.time()
    log_chatbot_response_time(end_time - start_time, source="챗봇")
    print(f"[-RAG-] run_llm_chain() result: {result}")
    return result["result"]

# 2025-08-11 감정 분석 함수 : 주어진 텍스트에 대해 감정 분석을 수행하는 함수
def analyze_sentiment(gender: str, age: str, emotion: str, meaning: str, action: str, reflect: str, anchor: str):
    chain = get_sentiment_chain()
    print(f"[-RAG-] analyze_sentiment() for emotional record")

    start_time = time.time()
    response = chain.run(
        gender=gender,
        age=age,
        emotion=emotion,
        meaning=meaning,
        action=action,
        reflect=reflect,
        anchor=anchor
    )
    end_time = time.time()
    log_chatbot_response_time(end_time - start_time, source="감정 분석")

    # LLM 응답을 파싱하여 감정 분류를 추출
    sentiment_class = "분류불가"
    if "긍정(Positive)" in response:
        sentiment_class = "긍정(Positive)"
    elif "부정(Negative)" in response:
        sentiment_class = "부정(Negative)"
    elif "중립(Neutral)" in response:
        sentiment_class = "중립(Neutral)"

    return response

# 감정 분석 스트리밍 함수 (RAG 기반 및 로깅 기능 추가, 2025-09-04 Gemini)
def analyze_sentiment_stream(config: dict, gender: str, age: str, emotion: str, meaning: str, action: str, reflect: str, anchor: str):
    """감정 분석을 스트리밍 방식으로 처리하고, 생성되는 텍스트 조각을 반환하며 응답 시간을 기록합니다."""
    start_time = time.time()
    chain = get_sentiment_chain(config)
    
    # 사용자 입력을 RAG 체인에 맞는 딕셔너리 형태로 구성, 2025-09-12 jylee
    input_data = {
        "gender": gender, "age": age, "emotion": emotion, "meaning": meaning,
        "action": action, "reflect": reflect, "anchor": anchor
    }
    # 사용자 입력 중 검색어(query)로 사용할 텍스트를 조합하여 딕셔너리에 추가, 2025-09-12 jylee
    query = f"{emotion} 감정의 이유: {meaning}, 오늘의 다짐: {anchor}"
    input_data["query"] = query
    print(f"--- [RAG Sentiment Analysis] Query: {query} ---")

    # 스트리밍 시작, 2025-09-12 jylee
    for chunk in chain.stream(input_data):
        yield chunk
    
    end_time = time.time()
    log_chatbot_response_time(end_time - start_time, source="감정 분석")


# 텍스트 요약 함수, 2025-08-19 jylee
def summarize_text(text_to_summarize: str) -> str:
    """
    Summarizes the given text using the LLM.
    """
    prompt = PromptTemplate(
        input_variables=["text_content"],
        template="다음 텍스트를 3~5문장으로 요약해 주세요:\n\n---\n{text_content}\n\n---\n\n요약:"
    )
    
    # chain = LLMChain(llm=models.get_llm(), prompt=prompt)
    chain = models.get_llm() | prompt  # 체인 연결 방식 변경
    
    # Take the first 1500 characters for a brief summary
    summary = chain.run(text_content=text_to_summarize[:1500])
    print(f"[-RAG-] Generated summary for text.")
    return summary
