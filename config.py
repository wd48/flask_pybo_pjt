import os

BASE_DIR = os.path.dirname(__file__)

SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR, 'pybo.db')) # 데이터베이스 접속 주소
SQLALCHEMY_TRACK_MODIFICATIONS = False # SQLAlchemy 이벤트를 처리하는 옵션 (pybo에서는 불필요하므로 비활성화)
SECRET_KEY = "dev"

# 이메일 설정
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USE_TLS = True
MAIL_USE_SSL = False
MAIL_USERNAME = 'kastrio.work@gmail.com'  # 실제 이메일로 변경 필요
MAIL_PASSWORD = 'cqhrxejbajsqodqb'      # 앱 비밀번호로 변경 필요
MAIL_DEFAULT_SENDER = 'kastrio.work@gmail.com'