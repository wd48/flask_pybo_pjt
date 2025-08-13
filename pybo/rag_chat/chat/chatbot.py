from langchain_community.chat_models import ChatOllama

# llm 전역변수로 선언
llm = None

# LLM 모델을 로드하는 함수
def load_llm(llm_model):
    global llm
    # ChatOllama 생성
    llm = ChatOllama(model=llm_model, temperature=0)