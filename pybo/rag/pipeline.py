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
from langchain.chains import LLMChain, RetrievalQA, create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
            llm=models.get_llm(),
            prompt=sentiment_prompt
        )
    print(f"[-RAG-] get_sentiment_chain() initialized with LLM: {current_app.config['LLM_MODEL']}")
    return sentiment_chain

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
    log_chatbot_response_time(end_time - start_time)
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
    log_chatbot_response_time(end_time - start_time)

    # LLM 응답을 파싱하여 감정 분류를 추출
    sentiment_class = "분류불가"
    if "긍정(Positive)" in response:
        sentiment_class = "긍정(Positive)"
    elif "부정(Negative)" in response:
        sentiment_class = "부정(Negative)"
    elif "중립(Neutral)" in response:
        sentiment_class = "중립(Neutral)"

    # 결과에서 감정 분류를 추출하여 로그
    # 예시로 'Positive', 'Negative', 'Neutral' 중 하나로 가정
    # LLM이 분류한 감정 클래스를 직접 파싱하는 로직이 필요
    # LLM 응답을 파싱하여 실제 감정 분류를 추출해야 함
    # 이 부분은 LLM의 응답 형식을 보고 적절히 파싱하여 수정해야 합니다.
    # log_sentiment_result(sentiment_class)

    return response

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
