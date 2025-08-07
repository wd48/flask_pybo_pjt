from datetime import datetime

from flask import Blueprint, render_template, request, url_for, g, flash
from werkzeug.utils import redirect

from .. import db
from pybo.models import Question
from pybo.forms import QuestionForm, AnswerForm
from pybo.views.auth_views import login_required

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
    
2025-08-05, 로그인 필요 함수 @login_required 데코레이터 추가
- 로그인하지 않은 사용자가 질문 등록을 시도하면 로그인 페이지로 리다이렉트.
- @login_required 데코레이터는 @bp.route 데코레이터보다 위에 위치할 경우 동작하지 않음
'''

# 데이터 전송 방식이 POST 인지 GET인지에 따라서 다르게 처리하는 부분이 코드의 핵심
@bp.route('/create/', methods=['GET', 'POST'])
@login_required
def create():
    form = QuestionForm()
    if request.method == 'POST' and form.validate_on_submit():
        question = Question(subject=form.subject.data, content=form.content.data, create_date=datetime.now(), user=g.user)
        db.session.add(question)
        db.session.commit()
        return redirect(url_for('main.index')) # 질문 목록 페이지로 리다이렉트
    return render_template('question/question_form.html', form=form)

# 2025-08-05, 질문 수정 기능 구현
# - 질문 수정은 질문 작성자만 가능하도록 구현
# - 질문 작성자가 아닌 사용자가 수정하려고 하면 '수정 권한이 없습니다.' 메시지를 출력하고 질문 상세 페이지로 리다이렉트
# - 질문 작성자가 수정하려고 하면 질문 폼에 기존 질문 데이터를 채워 넣어 수정할 수 있도록 함
@bp.route('/modify/<int:question_id>/', methods=['GET', 'POST'])
@login_required
def modify(question_id):
    question = Question.query.get_or_404(question_id)
    if g.user != question.user:
        flash('수정 권한이 없습니다.')
        return redirect(url_for('question.detail', question_id=question_id))
    if request.method == 'POST': # POST 요청인 경우
        form = QuestionForm()
        if form.validate_on_submit():
            form.populate_obj(question)
            question.modify_date = datetime.now()
            db.session.commit()
            return redirect(url_for('question.detail', question_id=question_id))
    else:   # GET 요청인 경우
        form = QuestionForm(obj=question) # 기존 질문 데이터를 폼에 채워 넣음
    return render_template('question/question_form.html', form=form)

# 2025-08-05, 질문 삭제 기능 구현
@bp.route('/delete/<int:question_id>')
@login_required
def delete(question_id):
    question = Question.query.get_or_404(question_id)

    if g.user != question.user:
        flash('삭제 권한이 없습니다.')
        return redirect(url_for('question.detail', question_id=question_id))
    db.session.delete(question)
    db.session.commit()
    return redirect(url_for('question._list'))  # 질문 목록 페이지로 리다이렉트

# 2025-08-07, 질문 추천 기능 구현
@bp.route('/vote/<int:question_id>')
@login_required
def vote(question_id):
    _question = Question.query.get_or_404(question_id)
    if g.user == _question.user:
        flash('자신의 질문에 추천할 수 없습니다.')
    else:
        _question.voter.append(g.user)
        db.session.commit()
    return redirect(url_for('question.detail', question_id=question_id))