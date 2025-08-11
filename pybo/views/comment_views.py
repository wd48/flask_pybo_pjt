from datetime import datetime

from flask import Blueprint, url_for, request, render_template, g, flash
from werkzeug.utils import redirect

from .. import db
from pybo.forms import CommentForm
from pybo.models import Question, Answer, Comment
from .auth_views import login_required

bp = Blueprint('comment', __name__, url_prefix='/comment')

@bp.route('/create/question/<int:question_id>/', methods=('POST',))
@login_required
def create_question_comment(question_id):
    form = CommentForm()
    question = Question.query.get_or_404(question_id)
    if form.validate_on_submit():
        comment = Comment(content=form.content.data, create_date=datetime.now(), user=g.user, question=question)
        db.session.add(comment)
        db.session.commit()
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(error)
    return redirect(url_for('question.detail', question_id=question_id))

@bp.route('/create/answer/<int:answer_id>/', methods=('POST',))
@login_required
def create_answer_comment(answer_id):
    form = CommentForm()
    answer = Answer.query.get_or_404(answer_id)
    if form.validate_on_submit():
        comment = Comment(content=form.content.data, create_date=datetime.now(), user=g.user, answer=answer)
        db.session.add(comment)
        db.session.commit()
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(error)
    return redirect(url_for('question.detail', question_id=answer.question.id))

@bp.route('/modify/<int:comment_id>/', methods=('GET', 'POST'))
@login_required
def modify(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    if g.user != comment.user:
        flash('수정 권한이 없습니다.')
        return redirect(_redirect_target(comment))
    if request.method == 'POST':
        form = CommentForm()
        if form.validate_on_submit():
            comment.content = form.content.data
            comment.modify_date = datetime.now()
            db.session.commit()
            return redirect(_redirect_target(comment))
    else:
        form = CommentForm(obj=comment)
    return render_template('comment/comment_form.html', form=form)

@bp.route('/delete/<int:comment_id>/', methods=('POST',))
@login_required
def delete(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    if g.user != comment.user:
        flash('삭제 권한이 없습니다.')
    else:
        db.session.delete(comment)
        db.session.commit()
    return redirect(_redirect_target(comment))


def _redirect_target(comment):
    if comment.question_id:
        return url_for('question.detail', question_id=comment.question_id)
    else:
        return url_for('question.detail', question_id=comment.answer.question.id)
