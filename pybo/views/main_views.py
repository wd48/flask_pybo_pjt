from flask import Blueprint, render_template, url_for
from werkzeug.utils import redirect

from pybo.models import Question

# 2025-07-23, 블루프린트 생성
bp = Blueprint('main', __name__, url_prefix='/')

# 2025-07-23, add a routing function to the main blueprint, 2xy
@bp.route('/hello')
def hello_pybo():
    return 'Hello, Pybo! from main_views.py'

# 함수명이 동일하면 에러나므로 주의
@bp.route('/')
def index():
    # # 2025-07-25 게시판 질문 목록 출력하기
    # # 게시판 질문 목록이 출력되도록 변경
    # question_list = Question.query.order_by(Question.create_date.desc())
    # return render_template('question/question_list.html', question_list=question_list)
    
    # 2025-07-25, 질문 목록 페이지로 리다이렉트
    # index 함수는 question._list 에 해당하는 URL로 리다이렉트
    # redirect : 입력받은 URL로 리다이렉트
    # url_for : 라우팅 함수명으로 URL을 역으로 찾는 함수
    return redirect(url_for('question._list'))

# 2025-07-25, 라우팅 함수 추가
# question_views.py 파일에 질문 목록, 질문 상세 기능을 구현했으므로
# main_views.py 파일에서는 해당 기능 제거
