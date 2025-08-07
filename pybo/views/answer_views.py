from datetime import datetime

from flask import Blueprint, url_for, request, render_template, g, flash
from werkzeug.utils import redirect

from pybo import db
from pybo.forms import AnswerForm
from pybo.models import Question, Answer
from .auth_views import login_required

bp = Blueprint('answer', __name__, url_prefix='/answer')

'''
create 함수의 매개변수 question_id는 URL 매핑 규칙을 통해 전달된다.
methods : 'POST'로 설정, 답변을 저장하는 질문 상세 템플릿의 form 속성이 POST 방식으로 같은 값을 지정해야 한다.
- 다른 폼 방식을 지정하면 'Method Not Allowed' 에러가 발생한다.
'''
# 답변 등록 라우팅 함수 수정하기
# POST 요청만 있으므로 분기처리 불필요
@bp.route('/create/<int:question_id>', methods=('POST',))
@login_required
def create(question_id):
    form = AnswerForm()
    question = Question.query.get_or_404(question_id)
    if form.validate_on_submit():
        content = request.form['content']
        answer = Answer(content=content, create_date=datetime.now(), user=g.user)
        question.answer_set.append(answer)
        db.session.commit()
        return redirect('{}#answer_{}'.format(
            url_for('question.detail', question_id=question_id), answer.id))
    return render_template('question/question_detail.html', question=question, form=form)

# 2025-08-05, 답변 수정 기능 구현
@bp.route('/modify/<int:answer_id>/', methods=('GET', 'POST'))
@login_required
def modify(answer_id):
    answer = Answer.query.get_or_404(answer_id)
    if g.user != answer.user:
        flash('수정 권한이 없습니다.')
        return redirect(url_for('question.detail', question_id=answer.question.id))
    if request.method == 'POST':
        form = AnswerForm()
        if form.validate_on_submit():
            form.populate_obj(answer)
            answer.modify_date = datetime.now() # 수정 날짜 업데이트
            db.session.commit()
            return redirect('{}#answer_{}'.format(
                url_for('question.detail', question_id=answer.question.id), answer.id))
    else: # GET 요청인 경우
        form = AnswerForm(obj=answer)
    return render_template('answer/answer_form.html', form=form)

# 2025-08-05, 답변 삭제 기능 구현
@bp.route('/delete/<int:answer_id>/', methods=('POST',))
@login_required
def delete(answer_id):
    answer = Answer.query.get_or_404(answer_id)
    question_id = answer.question.id
    if g.user != answer.user:
        flash('삭제 권한이 없습니다.')
    else:
        db.session.delete(answer)
        db.session.commit()
    return redirect(url_for('question.detail', question_id=question_id))

# 2025-08-05, 답변 추천 기능 구현
@bp.route('/vote/<int:answer_id>/')
@login_required
def vote(answer_id):
    _answer = Answer.query.get_or_404(answer_id)
    if g.user == _answer.user:
        flash('자신의 답변에 추천할 수 없습니다.')
    else:
        _answer.voter.append(g.user)
        db.session.commit()
    return redirect('{}#answer_{}'.format(url_for('question.detail', question_id=_answer.question.id), _answer.id))