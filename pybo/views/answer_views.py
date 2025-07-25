from datetime import datetime

from flask import Blueprint, url_for, request, render_template
from werkzeug.utils import redirect

from pybo import db
from pybo.forms import AnswerForm
from pybo.models import Question, Answer

bp = Blueprint('answer', __name__, url_prefix='/answer')

'''
create 함수의 매개변수 question_id는 URL 매핑 규칙을 통해 전달된다.
methods : 'POST'로 설정, 답변을 저장하는 질문 상세 템플릿의 form 속성이 POST 방식으로 같은 값을 지정해야 한다.
- 다른 폼 방식을 지정하면 'Method Not Allowed' 에러가 발생한다.
'''
# 답변 등록 라우팅 함수 수정하기
# POST 요청만 있으므로 분기처리 불필요
@bp.route('/create/<int:question_id>', methods=('POST',))
def create(question_id):
    form = AnswerForm()
    question = Question.query.get_or_404(question_id)
    if form.validate_on_submit():
        content = request.form['content']
        answer = Answer(content=content, create_date=datetime.now())
        question.answer_set.append(answer)
        db.session.commit()
        return redirect(url_for('question.detail', question_id=question_id))
    return render_template('question/question_detail.html', question=question, form=form)