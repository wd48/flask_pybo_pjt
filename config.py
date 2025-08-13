import os

BASE_DIR = os.path.dirname(__file__)

SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR, 'pybo.db')) # 데이터베이스 접속 주소
SQLALCHEMY_TRACK_MODIFICATIONS = False # SQLAlchemy 이벤트를 처리하는 옵션 (pybo에서는 불필요하므로 비활성화)
SECRET_KEY = "dev"

# 이메일 설정
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USE_TLS = True
MAIL_USE_SSL = False
MAIL_USERNAME = 'kastrio.work@gmail.com'  # 실제 이메일로 변경 필요
MAIL_PASSWORD = 'cqhrxejbajsqodqb'      # 앱 비밀번호로 변경 필요
MAIL_DEFAULT_SENDER = 'kastrio.work@gmail.com'

# 챗봇 관련 설정, 2025-08-12 jylee
CHAT_UPLOAD_FOLDER = 'uploads'  # 챗봇 업로드 폴더
CHAT_DB_PERSIST_DIR = 'chroma_db'   # ChromaDB 저장 폴더

# Embedding 모델 설정
EMBEDDING_MODEL = 'jhgan/ko-sroberta-multitask'

# LLM 설정
LLM_MODEL = 'gemma3n:latest'  # Ollama 모델 이름
LLM_TEMPERATURE = 0.7  # LLM 온도 설정

# 챗봇 업로드 폴더 설정
if not os.path.exists(CHAT_UPLOAD_FOLDER):
    os.makedirs(CHAT_UPLOAD_FOLDER)

CONFIG_DATA = None


######## 함수선언 ########
# 설정 데이터를 로드하는 함수
def set_retriever_config_data(search_type, key, value):
    # CONFIG_DATA가 None이거나 'retriever' 키가 없으면 파일에서 읽어옵니다
    if not CONFIG_DATA or 'retriever' not in CONFIG_DATA:
        print("warning: CONFIG_DATA not loaded, reading from file")
        return
    # 'retriever' 키가 없으면 초기화합니다
    if search_type in CONFIG_DATA['retriever']['search_types']:
        if key in CONFIG_DATA['retriever']['search_type'][search_type]:
            CONFIG_DATA['retriever']['search_type'][search_type][key] = value
            print(f"'{search_type}.{key}'설정이 '{value}'값으로 변경되었습니다")
        else:
            print(f"'{search_type}' search_type에 '{key}' 설정이 없습니다. '{value}'값으로 추가합니다")
    else:
        print(f"'{search_type}을 찾을 수 없습니다.")
