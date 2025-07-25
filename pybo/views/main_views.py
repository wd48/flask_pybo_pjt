from flask import Blueprint, render_template

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
    # 2025-07-25 게시판 질문 목록 출력하기
    # 게시판 질문 목록이 출력되도록 변경
    question_list = Question.query.order_by(Question.create_date.desc())
    return render_template('question/question_list.html', question_list=question_list)

# 2025-07-25, 라우팅 함수 추가
@bp.route('/detail/<int:question_id>/')
def detail(question_id):
    question = Question.query.get_or_404(question_id)
    return render_template('question/question_detail.html', question=question)