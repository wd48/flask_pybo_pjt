from datetime import datetime

from flask import Blueprint, render_template, request, url_for
from werkzeug.utils import redirect

from .. import db
from pybo.models import Question
from pybo.forms import QuestionForm, AnswerForm

bp = Blueprint('question', __name__, url_prefix='/question')

@bp.route('/list/')
def _list():
    page = request.args.get('page', type=int, default=1)  # 페이지 번호를 쿼리 파라미터에서 가져옴
    question_list = Question.query.order_by(Question.create_date.desc())
    question_list = question_list.paginate(page=page, per_page=10)  # 페이지네이션 처리
    return render_template('question/question_list.html', question_list=question_list)

@bp.route('/detail/<int:question_id>/')
def detail(question_id):
    form = AnswerForm()  # 답변 폼 생성
    question = Question.query.get_or_404(question_id)
    return render_template('question/question_detail.html', question=question, form=form)

'''
2025-07-25, 질문 등록 기능 구현
- request.method : creat 함수로 요청된 전송 방식
- form.validate_on_submit() : 전송된 폼 데이터의 정합성을 점검
    QuestionForm 클래스의 각 속성에 지정한 DataRequired() 같은 점검 항목에 이상이 없는지 확인
'''

# 데이터 전송 방식이 POST 인지 GET인지에 따라서 다르게 처리하는 부분이 코드의 핵심
@bp.route('/create/', methods=['GET', 'POST'])
def create():
    form = QuestionForm()
    if request.method == 'POST' and form.validate_on_submit():
        question = Question(subject=form.subject.data, content=form.content.data, create_date=datetime.now())
        db.session.add(question)
        db.session.commit()
        return redirect(url_for('main.index')) # 질문 목록 페이지로 리다이렉트
    return render_template('question/question_form.html', form=form)

