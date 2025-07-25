# 기술 스택 (ing)
- flask
- python 3.13.5
## 추가예정
- LangChain
- AI model : Gemma2 or Gemma 3n

# reference
- https://wikidocs.net/81044


## 플라스크 프로젝트 구조
    ├── pybo/
    │      ├─ __init__.py
    │      ├─ models.py
    │      ├─ forms.py
    │      ├─ views/
    │      │   └─ main_views.py
    │      ├─ static/
    │      │   └─ style.css
    │      └─ templates/
    │            └─ index.html
    └── config.py
  
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
## SQLite    
- 파이썬 기본 패키지에 포함, 소규모 프로젝트에서 사용하는 가벼운 파일을 기반으로 한 데이터베이스
- SQLite로 개발을 빠르게 진행하고 이후 운영 시스템 반영시에는 규모가 큰 데이터베이스로 교체함
