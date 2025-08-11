from flask_mail import Message
from pybo import mail
from flask import current_app
import threading

def send_async_email(app, msg):
    """비동기 이메일 발송"""
    with app.app_context():
        mail.send(msg)

def send_email(subject, recipients, text_body, html_body=None):
    """이메일 발송 함수"""
    msg = Message(
        subject=subject,
        recipients=recipients,
        body=text_body,
        html=html_body
    )

    # 비동기로 이메일 발송
    thread = threading.Thread(
        target=send_async_email,
        args=(current_app._get_current_object(), msg)
    )
    thread.start()

def send_password_reset_email(email, temp_password):
    """비밀번호 재설정 이메일 발송"""
    subject = '[2XY] 임시 비밀번호 발급'

    text_body = f'''
    안녕하세요,

    요청하신 임시 비밀번호가 발급되었습니다.
    
    임시 비밀번호: {temp_password}
    
    보안을 위해 로그인 후 비밀번호를 변경해 주세요.
    
    감사합니다.
    2XY 팀
    '''

    html_body = f'''
    <h3>임시 비밀번호 발급</h3>
    <p>안녕하세요,</p>
    <p>요청하신 임시 비밀번호가 발급되었습니다.</p>
    <p><strong>임시 비밀번호: {temp_password}</strong></p>
    <p>보안을 위해 로그인 후 <a href="#">비밀번호를 변경</a>해 주세요.</p>
    <p>감사합니다.<br>2XY 팀</p>
    '''

    send_email(subject, [email], text_body, html_body)
