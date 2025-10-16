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
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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

# 감정 분석 체인을 가져오는 함수 (RAG 기반 및 로깅 기능 추가, 2025-09-04 jylee)
def get_sentiment_chain(config: dict):
    """
    RAG 기반의 감정 분석 체인을 생성하고 반환합니다.
    전문가 지식 베이스에서 관련 정보를 검색하여, 이를 바탕으로 LLM이 답변을 생성합니다.
    """
    print("[-RAG-] (Sentiment Chain) Initializing...")
    # 1. 일관된 답변을 위해 전용 LLM 인스턴스 생성 (낮은 temperature)
    sentiment_llm = Ollama(
        base_url=config["LLM_HOST"],
        model=config["LLM_MODEL"],
        temperature=0.1
    )

    # 2. 모든 지식 베이스 컬렉션에서 통합 검색을 위한 Retriever 가져오기, 2025-10-10 jylee
    # 여러 KB 파일에 대한 동시 검색을 지원합니다.
    # 참고: vectorstore.py에 get_all_kb_collections() 함수가 필요할 수 있습니다.
    all_kb_collections = vectorstore.get_all_kb_collections()

    retriever = None
    if all_kb_collections:
        # 리트리버 설정, k=3
        retrievers = [
            collection['vectordb'].as_retriever(search_kwargs={"k": 3})
            for collection in all_kb_collections.values()
        ]
        
        if len(retrievers) > 1:
            # 2개 이상의 KB가 있으면 EnsembleRetriever로 결합
            collection_names = [c['collection_name'] for c in all_kb_collections.values()]
            print(f"[-RAG-] (Sentiment Chain) Creating EnsembleRetriever for {len(retrievers)} KB collections: {collection_names}")
            retriever = EnsembleRetriever(retrievers=retrievers, weights=[0.5] * len(retrievers))
        elif len(retrievers) == 1:
            # 1개의 KB만 있으면 해당 retriever를 바로 사용
            collection_name = list(all_kb_collections.values())[0]['collection_name']
            print(f"[-RAG-] (Sentiment Chain) Using single retriever for KB collection: '{collection_name}'")
            retriever = retrievers[0]

    # 3. 새로운 RAG 프롬프트 템플릿
    sentiment_prompt = PromptTemplate(
        input_variables=["context", "gender", "age", "emotion", "meaning", "action", "reflect", "anchor"],
        template="""
            당신은 인지행동치료(CBT)와 긍정심리학에 기반한 전문 심리 상담가입니다.
            사용자의 감정 기록을 분석하고, 공감적이고 실용적인 조언을 제공해야 합니다.
            반드시 아래의 형식과 지침에 따라 답변을 생성해 주세요.

            ---
            ### 지침

            1.  **[전문가 조언] 활용**: 주어진 [전문가 조언]이 있다면, 반드시 그 내용을 **행동 분석**과 **심층 제언**에 적극적으로 참고하고 인용하여 답변을 작성하세요.
            2.  **조언이 없는 경우**: [전문가 조언]에 "참고할 전문가 조언을 찾지 못했습니다."라고 표시되면, 당신의 일반적인 심리학 전문 지식을 바탕으로 답변을 생성하세요.
            3.  **형식 준수**: 아래의 "분석 답변 형식"에 있는 모든 항목(감정 진단, 행동 분석, 심층 제언, 마음 다지기)을 반드시 포함하고, 마크다운(`**`)을 사용하여 소제목을 강조해 주세요.

            ---
            ### 전문가 조언

            {context}

            ---
            ### 감정 기록

            - **성별**: {gender}
            - **연령대**: {age}
            - **감정**: {emotion}
            - **이유**: {meaning}
            - **도움이 된 행동**: {action}
            - **긍정적 변화**: {reflect}
            - **오늘의 다짐**: {anchor}

            ---
            ### 분석 답변 형식

            **1. 감정 진단**
            사용자의 감정과 그 이유를 바탕으로, 현재 감정 상태를 1~2문장으로 명확하게 진단합니다.
            *예: "오늘 느끼신 {emotion}은(는) {meaning}에서 비롯된 자연스러운 반응으로 보입니다."*

            **2. 행동 분석**
            사용자가 시도한 '도움이 된 행동'이 심리학적으로 어떤 의미가 있고, 왜 '긍정적 변화'를 가져왔는지 분석합니다. [전문가 조언]이 있다면, 그 이론을 근거로 들어 설명합니다.
            *예: "{action}은(는) [전문가 조언 또는 심리학 이론]에 따르면 '행동 활성화' 기법에 해당하며, 이는 무기력감을 줄이고 긍정적인 감정을 유도하는 데 매우 효과적입니다."*

            **3. 심층 제언**
            사용자의 상황에 맞춰 시도해볼 수 있는 구체적인 추가 활동이나 생각의 전환 방법을 **반드시 2가지** 제안합니다. 각 제언은 "기대 효과"를 포함해야 합니다.
            - **제언 1**: [구체적인 활동 제안]
              - **기대 효과**: [제안된 활동이 심리적으로 어떤 도움을 주는지에 대한 설명]
            - **제언 2**: [다른 활동 또는 생각 전환 방법 제안]
              - **기대 효과**: [이것이 왜 도움이 되는지에 대한 설명]

            **4. 마음 다지기**
            사용자의 '오늘의 다짐'을 인용하며, 긍정적인 지지와 격려의 메시지로 마무리합니다.
            *예: "{anchor}라고 다짐하신 것처럼, 오늘의 소중한 경험이 앞으로 나아가는 데 훌륭한 밑거름이 될 것입니다."*
        """
    )

    # 4. RAG 체인 구성 (LCEL 방식) 및 로깅 추가
    def log_and_format_docs(docs):
        print("\n--- [-RAG-] (Sentiment Chain) Retrieved Context: ---")
        if not docs:
            print("[-RAG-] (Sentiment Chain) No relevant context found in Knowledge Base.")
            return "참고할 전문가 조언을 찾지 못했습니다."
        
        for doc in docs:
            source = doc.metadata.get('filename', 'N/A')
            content_preview = doc.page_content.replace('\n', ' ')[:100]
            print(f"  - SOURCE: {source}")
            print(f"    CONTENT: {content_preview}...\n")
        print("-----------------------------------------------------\
")
        return "\n\n".join(doc.page_content for doc in docs)

    # retriever 존재 여부에 따라 동적으로 context 생성 체인 구성, 2025-10-15 jylee
    if retriever:
        context_chain = itemgetter("query") | retriever | log_and_format_docs
    else:
        # retriever가 없으면 (KB가 없으면) 기본 메시지 반환
        print("[-RAG-] (Sentiment Chain) No KB collections found. Retriever will be skipped.")
        print("\n--- [RAG Sentiment Analysis] No retriever available. Skipping context search. ---")
        context_chain = RunnableLambda(lambda x: "참고할 전문가 조언을 찾지 못했습니다.")

    # 최종 답변을 생성하는 LLM 체인
    llm_chain = sentiment_prompt | sentiment_llm | StrOutputParser()

    # context와 최종 answer를 모두 반환하도록 체인 재구성
    rag_chain = RunnablePassthrough.assign(
        context=context_chain
    ).assign(
        answer=llm_chain
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

# 감정 분석 스트리밍 함수 (RAG 기반 및 로깅 기능 추가, 2025-09-04 jylee)
def analyze_sentiment_stream(config: dict, gender: str, age: str, emotion: str, meaning: str, action: str, reflect: str, anchor: str):
    """감정 분석을 스트리밍 방식으로 처리하고, 생성되는 텍스트 조각을 반환하며 응답 시간을 기록합니다."""
    start_time = time.time()
    
    # RAG 체인 및 관련 구성 요소를 가져옵니다.
    chain = get_sentiment_chain(config)
    
    # 사용자 입력을 RAG 체인에 맞는 딕셔너리 형태로 구성
    input_data = {
        "gender": gender, "age": age, "emotion": emotion, "meaning": meaning,
        "action": action, "reflect": reflect, "anchor": anchor,
        "query": f"{emotion} 감정의 이유: {meaning}, 오늘의 다짐: {anchor}"
    }
    print(f"--- [RAG Sentiment Analysis] Query: {input_data['query']} ---")

    # 1. 먼저 context를 생성하고 즉시 반환합니다.
    context = chain.invoke(input_data)['context']
    yield {"context": context}

    # 2. context를 포함한 전체 입력을 사용하여 답변 생성을 스트리밍합니다.
    answer_chain = chain.pick("answer")
    for chunk in answer_chain.stream(input_data):
        yield {"answer": chunk}
    
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
