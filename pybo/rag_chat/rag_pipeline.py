# pybo/rag_chat/rag_pipeline.py
import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

load_dotenv()
model_path = os.getenv("DATA_FILE_PATH")
# 1. 임베딩 모델
embedding_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    cache_folder=model_path
)

# 2. Chroma 벡터 DB 로드 또는 생성
CHROMA_PATH = "./chroma_db"
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

vectordb = Chroma(
    collection_name="rag_kochat",
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH
)

# 3. LLM - Gemma 2b
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=token)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    {context}

    Question: {question}
    Answer:
    """
)

# 2025-08-04 ollama로 변경
llm = Ollama(model="gemma2:latest")

# 4. RAG 체인
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt}
)

# 5. 질의 함수
def ask_rag(query: str):
    result = qa_chain.invoke(query)
    if isinstance(result, dict) and 'result' in result:
        answer_text = result['result']
        if 'Answer:' in answer_text:
            return answer_text.split('Answer:')[1].strip()
        return answer_text.strip()
    return result