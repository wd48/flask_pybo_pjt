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

## 쿠키와 세션
- 웹 프로그램 : 웹 프라우저 요청 -> 서버 응답 순서로 실행되며, 서버 응답이 완료되면 웹 브라우저와 서버 사이의 네트워크 연결은 끊어진다
- 동일한 브라우저의 요청에서 서버는 동일한 세셩은 사용한다
- 쿠키 : 서버가 웹 브라우저에 발행하는 값으로, 웹 브라우저가 서버에 어떤 요청을 하면 서버는 쿠키를 생성하여 전송하는 방식으로 응답한다. 웹 브라우저는 서버에서 받은 쿠키를 저장한다.   
  - 이 후 서버에 다시 오청을 보낼 때는 저장한 쿠키를 HTTP 헤더에 담아 전송한다.
  - 그러면 서버는 웹 브라우저가 보낸 쿠키를 이전에 발행했던 쿠키값과 비교하여 같은 웹 브라우저에서 요청한 것인지 아닌지를 구분할 수 있다.
- 이 때 세션은 쿠키 1개당 생성되는 서버의 메모리 공간이라고 할 수 있다.