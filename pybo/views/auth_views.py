from flask import Blueprint, url_for, render_template, flash, request, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import redirect
import functools
import secrets
import string

from pybo import db
from pybo.forms import UserCreateForm, UserLoginForm, PasswordResetRequestForm, PasswordChangeForm
from pybo.models import User
from pybo.email_utils import send_password_reset_email

bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/signup/', methods=('GET', 'POST'))
def signup():
    form = UserCreateForm()
    if request.method == 'POST' and form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None:
            user = User(username=form.username.data,
                        password=generate_password_hash(form.password1.data),
                        email=form.email.data)
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('main.index'))
        else:
            flash('이미 존재하는 사용자입니다.')
    return render_template('auth/signup.html', form=form)

# 사용자 로그인 라우팅 함수
'''
로그인 과정
- 폼 입력으로 받은 username으로 DB에 해당 사용자가 있는지 검사
- 사용자가 없으면 오류 발생
- 사용자 존재하면 폼 입력으로 받은 password와 check_password_hash() 함수를 이용해 비밀번호를 검사
'''
@bp.route('/login/', methods=('GET', 'POST'))
def login():
    form = UserLoginForm()
    if request.method == 'POST' and form.validate_on_submit():
        error = None
        user = User.query.filter_by(username=form.username.data).first()
        if not user:
            error = '존재하지 않는 사용자입니다.'
        elif not check_password_hash(user.password, form.password.data):
            error = '비밀번호가 일치하지 않습니다.'
        if error is None:
            session.clear()
            session['user_id'] = user.id
            # 2025-08-05, 로그인 성공 후 리다이렉트할 URL을 처리
            _next = request.args.get('next','')
            if _next:
                return redirect(_next)
            else:
                return redirect(url_for('main.index'))
        flash(error)
    return render_template('auth/login.html', form=form)

# 로그인 여부 확인
# auth_views.py의 라우팅 함수 뿐만 아니라 모든 라우팅 함수보다 항상 먼저 실행된다
@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = User.query.get(user_id)

# 로그아웃 라우팅 함수
@bp.route('/logout/')
def logout():
    session.clear()
    return redirect(url_for('main.index'))

# 2025-08-05
# g.user가 있는지 조사하여 없으면 로그인 URL로 리다이렉트하는 데코레이터 함수
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(*arg, **kwargs):
        if g.user is None:
            _next = request.url if request.method == 'GET' else ''
            return redirect(url_for('auth.login', next=_next))
        return view(*arg, **kwargs)
    return wrapped_view

# 2025-08-11 jylee, 비밀번호 찾기 - 임시 비밀번호 이메일 발송
@bp.route('/reset_password/', methods=('GET', 'POST'))
def reset_password():
    form = PasswordResetRequestForm()
    if request.method == 'POST' and form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            # 임시 비밀번호 생성 (8자리 영문+숫자)
            temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))

            # 임시 비밀번호로 변경
            user.password = generate_password_hash(temp_password)
            db.session.commit()

            # 이메일 발송
            try:
                send_password_reset_email(user.email, temp_password)
                flash('임시 비밀번호가 이메일로 발송되었습니다. 이메일을 확인해주세요.')
            except Exception as e:
                # 이메일 발송 실패 시 임시 비밀번호를 화면에 표시 (개발/테스트용)
                flash(f'이메일 발송에 실패했습니다. 임시 비밀번호: {temp_password}')
            return redirect(url_for('auth.login'))
        else:
            flash('등록되지 않은 이메일 주소입니다.')
    return render_template('auth/reset_password.html', form=form)

# 2025-08-11 jylee 비밀번호 변경
@bp.route('/change_password/', methods=('GET', 'POST'))
@login_required
def change_password():
    form = PasswordChangeForm()
    if request.method == 'POST' and form.validate_on_submit():
        # 현재 비밀번호 확인
        if check_password_hash(g.user.password, form.current_password.data):
            # 새 비밀번호로 변경
            g.user.password = generate_password_hash(form.new_password1.data)
            db.session.commit()
            flash('비밀번호가 성공적으로 변경되었습니다.')
            return redirect(url_for('main.index'))
        else:
            flash('현재 비밀번호가 일치하지 않습니다.')
    return render_template('auth/change_password.html', form=form)
