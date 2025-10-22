# Pybo: AI-Powered Q&A and Emotional Analysis Platform

## 1. 개요 (Overview)

본 프로젝트는 Python Flask 프레임워크를 기반으로 구축된 웹 애플리케이션입니다. 전통적인 Q&A 게시판 기능에 더하여, **RAG(Retrieval-Augmented Generation, 검색 증강 생성)** 기술을 활용한 두 가지 핵심 AI 기능을 제공합니다.

1.  **문서 기반 RAG 챗봇**: 사용자가 업로드한 PDF 문서의 내용을 기반으로 질문에 답변하는 대화형 챗봇입니다.
2.  **전문가 지식 기반 감정 분석**: 사용자가 기록한 감정을 심리학 전문가의 지식 베이스를 참고하여 깊이 있게 분석하고 조언하는 서비스입니다.

이 외에도 답변의 성능을 측정하고 시각화하는 대시보드 기능을 포함하여 AI 모델의 응답을 지속적으로 평가하고 관리할 수 있습니다.

## 2. 주요 기능 (Key Features)

### 2.1. 전통적인 Q&A 게시판
- **게시글 관리**: 질문과 답변의 생성(Create), 조회(Read), 수정(Update), 삭제(Delete) (CRUD).
- **사용자 상호작용**: 질문/답변에 대한 추천 및 댓글 작성 기능.
- **편의 기능**: 게시글 검색, 최신순/추천순 정렬, 페이지네이션, 조회수 표시.
- **사용자 인증**: 회원가입, 로그인/로그아웃, 비밀번호 변경 및 이메일을 통한 임시 비밀번호 발급.

### 2.2. 문서 기반 RAG 챗봇
- **문서 기반 질의응답**: 사용자가 업로드한 PDF 파일의 내용을 근거로 AI가 질문에 답변합니다.
- **문서 관리**: 사용자는 PDF 파일을 업로드, 조회, 삭제할 수 있습니다. 각 파일은 독립된 데이터베이스(Vector Collection)로 관리되어 문서 간 데이터가 섞이지 않습니다.
- **대화형 인터페이스**: 채팅 기록을 기억하여 이어지는 질문에도 자연스럽게 답변합니다.

### 2.3. 전문가 지식 기반 감정 분석
- **감정 기록**: 사용자는 자신의 성별, 연령, 감정, 행동 등을 정해진 양식에 따라 기록합니다.
- **RAG 기반 분석**: 단순한 LLM의 답변이 아닌, 관리자가 업로드한 **전문가 지식 베이스(심리학, 정신 건강 관련 PDF/TXT)**의 내용을 검색하고, 이를 근거로 LLM이 전문적인 분석과 조언을 생성합니다.
- **지식 베이스 관리**: 관리자는 감정 분석의 근거가 될 전문가 문서를 별도로 업로드하고 관리할 수 있습니다.

### 2.4. 성능 및 평가 대시보드
- **응답 시간 시각화**: 챗봇과 감정 분석 기능의 응답 시간 추이를 차트로 시각화하여 성능을 모니터링합니다.
- **답변 평가**: AI가 생성한 모든 답변(챗봇, 감정 분석)은 백그라운드에서 **관련성(Relevance), 간결성(Conciseness)** 등의 기준으로 자동 평가됩니다.
- **평가 결과 조회**: 평가 로그를 `normal_rag`(일반 챗봇)와 `sentiment_rag`(감정 분석)로 나누어 조회하고, 각 답변의 품질을 검토할 수 있습니다.

## 3. 아키텍처 및 기술 스택 (Architecture & Tech Stack)

이 애플리케이션은 웹 서버와 AI 모델 서버가 분리된 현대적인 아키텍처를 따릅니다.

- **Web Framework**: `Flask`
- **Database**: `SQLAlchemy` (ORM), `Alembic` (Migration), `SQLite` (Default)
- **AI Framework**: `LangChain` (특히 LCEL을 활용한 파이프라인 구성)
- **Large Language Model (LLM)**: `Ollama` (외부 서버로 실행, `gemma2`, `exaone` 등)
- **Embedding Model**: `jhgan/ko-sroberta-multitask` (Flask 앱에서 로컬로 로드)
- **Vector Database**: `ChromaDB` (외부 서버, Docker로 실행)
- **Frontend**: `Jinja2`, `Bootstrap 5`
- **Evaluation**: `LangChain Evaluation`, `Langfuse` (검토 중)

---

## 4. 데이터 흐름 (Data Flow)

### 4.1. RAG 챗봇 데이터 흐름
1.  **[준비] 문서 업로드**: 사용자가 '문서 관리' 페이지에서 PDF 파일을 업로드합니다.
2.  **[준비] 인덱싱**: `upload_utils.py`가 파일을 받아 청크(Chunk)로 분할하고, `jhgan_ko-sroberta-multitask` 임베딩 모델을 사용해 각 청크를 벡터로 변환합니다.
3.  **[준비] DB 저장**: 변환된 벡터는 파일별로 생성된 고유한 컬렉션에 ChromaDB로 저장됩니다.
4.  **[실행] 질문 입력**: 사용자가 챗봇 UI에서 질문을 입력합니다.
5.  **[실행] 검색(Retrieve)**: `pipeline.py`의 `get_conversational_rag_chain`이 실행됩니다. 사용자의 질문을 벡터로 변환한 뒤, ChromaDB에서 의미적으로 가장 유사한 문서 청크(Context)를 검색합니다.
6.  **[실행] 생성(Generate)**: 검색된 Context와 사용자 질문을 프롬프트로 조합하여 Ollama LLM 서버에 전달합니다.
7.  **[실행] 답변**: LLM이 생성한 답변을 사용자 화면에 표시합니다.

### 4.2. 감정 분석 데이터 흐름
1.  **[준비] 지식 베이스 업로드**: 관리자가 '지식 베이스 관리' 페이지에서 전문가 문서(PDF, TXT)를 업로드합니다. 모든 문서는 `kb_` 접두사가 붙은 고유한 컬렉션에 저장됩니다.
2.  **[실행] 감정 기록 제출**: 사용자가 '감정 분석' 페이지에서 자신의 상태를 기록하고 '분석' 버튼을 누릅니다.
3.  **[실행] RAG 체인 실행**: `pipeline.py`의 `get_sentiment_chain` 함수가 실행됩니다.
    a. **검색(Retrieve)**: 사용자의 '감정 이유' 텍스트를 검색어로 삼아, **모든 `kb_` 컬렉션**에서 가장 관련성 높은 전문가 조언(Context)을 검색합니다.
    b. **프롬프트 구성**: 검색된 Context와 사용자의 전체 감정 기록을 상세한 프롬프트 템플릿에 결합합니다.
    c. **생성(Generate)**: 완성된 프롬프트를 Ollama LLM 서버에 전달하여 전문적인 분석 답변을 생성합니다.
4.  **[실행] 답변 스트리밍**: 생성된 답변은 실시간으로 사용자 화면에 표시됩니다.

---

## 5. 사용 방법 (Getting Started)

### 5.1. 사전 요구사항
- Python 3.10+
- Docker Desktop
- Ollama 및 원하는 LLM 모델 (예: `ollama pull gemma2`)

### 5.2. 설치 및 실행
1.  **저장소 복제**:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **가상 환경 및 의존성 설치**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **환경 변수 설정**:
    프로젝트 루트에 `.env` 파일을 생성하고, `config.py`를 참고하여 필요한 환경 변수를 설정합니다.
    ```env
    FLASK_APP=pybo
    FLASK_ENV=development
    DATABASE_URL=sqlite:///pybo.db

    # Ollama 서버 주소
    LLM_HOST=http://localhost:11434
    LLM_MODEL=gemma2
    LLM_TEMPERATURE=0.1

    # ChromaDB 서버 주소
    CHROMA_HOST=localhost
    CHROMA_PORT=8000
    ```

4.  **ChromaDB 서버 실행 (Docker)**:
    ```bash
    docker run -d -p 8000:8000 --name chroma-db -v "%cd%/chroma_data:/chroma/.chroma/server/data" chromadb/chroma
    ```
    *`%cd%`는 Windows 기준이며, macOS/Linux에서는 `$(pwd)`를 사용하세요.*

5.  **Ollama LLM 서버 실행**:
    별도의 터미널에서 Ollama 서버를 실행합니다.
    ```bash
    ollama serve
    ```

6.  **데이터베이스 초기화**:
    ```bash
    flask db upgrade
    ```

7.  **애플리케이션 실행**:
    ```bash
    flask run
    ```

8.  **접속**: 웹 브라우저에서 `http://127.0.0.1:5000`으로 접속합니다.

### 5.3. 주요 기능 사용법
- **Q&A 게시판**: `/question/list` 경로에서 질문을 작성하고 답변을 달 수 있습니다.
- **RAG 챗봇**:
    1. `/chat/files`에서 PDF 문서를 업로드합니다.
    2. `/chat` 페이지로 이동하여, 우측 '문서 선택' 드롭다운에서 원하는 문서를 선택하고 질문합니다.
- **감정 분석**:
    1. (최초 1회) `/chat/kb`에서 감정 분석에 참고할 전문가 문서를 업로드합니다.
    2. `/chat/sentiment_analysis` 페이지에서 양식을 작성하고 '분석' 버튼을 누릅니다.
- **평가 결과**: `/chat/evaluation_results`에서 AI가 생성한 모든 답변에 대한 자동 평가 결과를 확인할 수 있습니다.

## 6. 프로젝트 구조

```text
/
├─ pybo/           # Flask 애플리케이션 모듈
│  ├─ rag/         # RAG 챗봇 및 감정 분석 기능 관련 모듈
│  ├─ static/      # CSS, JS, 이미지 등 정적 파일
│  ├─ templates/   # HTML 템플릿 파일
│  ├─ views/       # Q&A 게시판 등 기본 기능 뷰
│  └─ __init__.py  # 애플리케이션 팩토리
├─ logs/           # AI 답변 평가 로그 저장
│  ├─ normal_rag/
│  └─ sentiment_rag/
├─ migrations/     # Alembic 데이터베이스 마이그레이션
├─ uploads/        # RAG 챗봇용 PDF 업로드
│  └─ sentiment_kb/ # 감정 분석용 지식 베이스 문서 업로드
├─ .env            # 환경 변수 설정 파일
├─ config.py       # Flask 설정 파일
├─ requirements.txt # Python 의존성 목록
└─ README.md       # 프로젝트 설명서
```

## 7. 참고 자료
- [Flask Tutorial](https://wikidocs.net/81044)
- [LangChain Tutorial](https://wikidocs.net/233351)
- [LangChain Evaluation](https://python.langchain.com/api_reference/langchain/evaluation.html)
- [LangFuse](https://langfuse.com/kr)