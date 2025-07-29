# 기술 스택 (ing)
- flask
- python 3.13.5
## 추가예정
- LangChain
- models
  - embedding : jhgan/ko-sroberta-multitask
  - LLM : google/gemma-3n-it

# reference
- https://wikidocs.net/81044
---
### 실행방법
```bash
# 가상환경 및 필요 유틸 설치
# flask 설치
$ pip install flask

# pip 업데이트 요청 시
$ python -m pip install --upgrade pip

# ORM 라이브러리 설치
pip install flask-migrate


# 1. 가상환경 활성화 및 패키지 설치: venv2 가상환경을 활성화합니다.
  # Windows
  $ venv2\Scripts\activate.bat
  # Linux/Mac
  $ source venv2/bin/activate
  
  # 패키지 설치 : requirements.txt 파일에 정의된 패키지를 설치합니다.
  $ pip install -r resources/requirements.txt
    
# 2. 데이터베이스 마이그레이션 적용 (최초 1회)
  Alembic을 사용해 데이터베이스를 초기화합니다.
  flask db upgrade
  
# 3. HuggingFace 모델 다운로드 (최초 1회)
  # HuggingFace에서 필요한 모델을 다운로드합니다.
  # 아래 명령어를 실행하기 전에 먼저 huggingface_hub 패키지를 설치하고 HuggingFace 계정으로 로그인해야 합니다.
  $ pip install huggingface_hub
  $ huggingface-cli login
  $ hf download jhgan/ko-sroberta-multitask --local-dir resources/models/ko-sroberta
  $ hf download google/gemma-3n-E2B-it --local-dir resources/models/gemma-3n-E2B-it

# 4.앱 실행 : Flask 앱을 실행합니다.
  $ flask run
  # 또는
  $ python -m flask ru

5. 웹 브라우저에서 접속
  # 기본적으로 http://127.0.0.1:5000 에서 앱에 접속할 수 있습니다.

## 환경 변수(FLASK_APP, FLASK_ENV)가 필요하다면 아래처럼 설정합니다.
$ set FLASK_APP=pybo
$ set FLASK_ENV=development
```

```bash
# 데이터베이스 초기 파일 생성 (venv 환경에서 실행)
# echo $FLASK_APP 으로 프로젝트 메인 py 파일명이 나오는지 확인한다
$ flask db init

# 모델 생성 및 변경 시 사용 (작업파일이 생성됨)
$ flask db migrate

# 모델 변경내용을 실제 DB에 적용할떄 사용 (생성된 작업파일을 실행, DB를 변경함)
$ flask db upgrade

# 플라스크 셀 : db 조회 쿼리 입력,수정,삭제 등을 python 방식으로 가능하게 함
# 종료 : Ctrl+z > Enter or exit()
$ flask shell
```

---
## 플라스크+LangChain with RAG 프로젝트 구조
```text
/
├── config.py
├── pybo.db
├── README.md
├── structure.txt
├── chroma_db/
├── migrations/
├── pybo/
│   ├── __init__.py
│   ├── filter.py
│   ├── forms.py
│   ├── models.py
│   ├── rag_chat/
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py
│   │   ├── routes.py
│   ├── resources/
│   │   └── upload/
│   ├── static/
│   │   ├── bootstrap.min.css
│   │   ├── bootstrap.min.js
│   │   └── style.css
│   ├── templates/
│   │   ├── base.html
│   │   ├── form_errors.html
│   │   ├── navbar.html
│   │   ├── auth/
│   │   ├── question/
│   │   └── rag_chat/
│   └── views/
│       ├── answer_views.py
│       ├── auth_views.py
│       ├── main_views.py
│       ├── question_views.py
├── resources/
│   ├── bootstrap-5.1.3-dist.zip
│   ├── requirements_250729.txt
│   ├── requirements.txt
│   └── models/
│       ├── gemma-3n-E2B-it/
│       └── ko-sroberta/
├── shell/
│   └── kindlab_pjt.cmd
├── vector_store/
│   └── chroma.sqlite3
├── venv2/
```
  
- pybo 디렉터리 안에 있는 __init__.py 파일이 pybo.py 파일의 역할을 대신할 것

**1. models.py : 데이터베이스 처리**    
- ORM(object relational mapping)을 지원하는 파이썬 데이터베이스 도구인 SQLAlchemy 사용.
- "모델 클래스들을 정의할 models.py 파일이 필요하다"

**2. forms.py : 서버로 전송된 폼 처리**    
- WTForms 라이브러리 : 웹 브라우저에서 서버로 전송된 폼을 처리할 때 사용
- WTForms 역시 모델 기반으로 폼을 처리한다. 그래서 폼 클래스를 정의할 forms.py 파일 필요

**3. views : 화면 구성용 디렉토리**    
- pybo.py 파일에 작성했던 함수들로 구성된 뷰 파일들을 저장
- 기능에 따라 main_views.py, question_views.py, answer_views.py 등의 뷰 파일을 만들 것

**4. static : CSS, 자바스크립트, 이미지 파일을 저장하는 디렉터리**    
- 프로젝트의 스타일시트(.css), 자바스크립트(.js) 그리고 이미지 파일(.jpg, .png) 등을 저장

**5. templates : HTML 파일을 저장하는 디렉터리**    
- templates 디렉터리에는 파이보의 질문 목록, 질문 상세 등의 HTML 파일을 저장한다. 파이보 프로젝트는 question_list.html, question_detail.html과 같은 템플릿 파일을 만들어 사용할 것이다.

**6. config.py : 프로젝트 설정 파일**    
- 프로젝트의 환경변수, 데이터베이스 등의 설정 저장
---
## SQLite    
- 파이썬 기본 패키지에 포함, 소규모 프로젝트에서 사용하는 가벼운 파일을 기반으로 한 데이터베이스
- SQLite로 개발을 빠르게 진행하고 이후 운영 시스템 반영시에는 규모가 큰 데이터베이스로 교체함

## 쿠키와 세션
- 웹 프로그램 : 웹 프라우저 요청 -> 서버 응답 순서로 실행되며, 서버 응답이 완료되면 웹 브라우저와 서버 사이의 네트워크 연결은 끊어진다
- 동일한 브라우저의 요청에서 서버는 동일한 세셩은 사용한다
- 쿠키 : 서버가 웹 브라우저에 발행하는 값으로, 웹 브라우저가 서버에 어떤 요청을 하면 서버는 쿠키를 생성하여 전송하는 방식으로 응답한다. 웹 브라우저는 서버에서 받은 쿠키를 저장한다.   
  - 이 후 서버에 다시 오청을 보낼 때는 저장한 쿠키를 HTTP 헤더에 담아 전송한다.
  - 그러면 서버는 웹 브라우저가 보낸 쿠키를 이전에 발행했던 쿠키값과 비교하여 같은 웹 브라우저에서 요청한 것인지 아닌지를 구분할 수 있다.
- 이 때 세션은 쿠키 1개당 생성되는 서버의 메모리 공간이라고 할 수 있다.

---
## HuggingFace 모델 로컬 다운로드
```bash
# HuggingFace 모델을 로컬에 다운로드하여 사용할 수 있습니다.
# 아래 명령어를 실행하기 전에 먼저 huggingface_hub 패키지를 설치하고
# HuggingFace 계정으로 로그인해야 합니다. 
# + venv 환경 활성화 필요

# 1. huggingface-cli 설치
pip install huggingface_hub

# 2. 로그인 (최초 한 번)
huggingface-cli login

# 3. 모델 다운로드
# embedding 모델 : jhgan/ko-sroberta-multitask
# LLM 모델 : google/gemma-3n-E2B-it
# 다운로드한 모델은 .resources/models/ko-sroberta 및 .resources/models/gemma-3n-E2B-it 디렉터리에 저장됩니다.
hf download jhgan/ko-sroberta-multitask --local-dir .resources/models/ko-sroberta
hf download google/gemma-3n-E2B-it --local-dir .resources/models/gemma-3n-E2B-it
```