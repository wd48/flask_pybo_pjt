from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData
from flask_mail import Mail

import config
'''
2025-07-25
- db 객체를 create_app 함수 안에서 생성하면 블루프린트와 같은 모듈에서 db 객체를 사용할 수 없으므로
db, migrate 객체를 모듈 레벨에서 생성하고 (create_app 함수 밖), create_app 함수에서 초기화하는 방식으로 변경합니다.
- 해당 객체 (db, migrate)를 앱에 등록할 때는 create_app 함수에서 init_app 함수를 통해 진행한다.
'''

naming_convention = {
    "ix": 'ix_%(column_0_label)s',
    "uq": 'uq_%(table_name)s_%(column_0_name)s',
    "ck": 'ck_%(table_name)s_%(constraint_name)s',
    "fk": 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s',
    "pk": 'pk_%(table_name)s'
}
db = SQLAlchemy(metadata=MetaData(naming_convention=naming_convention))
migrate = Migrate()
mail = Mail()
from . import models  # 모델을 임포트하여 SQLAlchemy가 모델 클래스를 인식하도록 함

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    # RAG 파이프라인 모델 사전 로딩
    with app.app_context():
        from .rag import models
        models.init_models()

    # ORM
    db.init_app(app)
    mail.init_app(app)

    if app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite'):
        migrate.init_app(app, db, render_as_batch=True)
    else:
        migrate.init_app(app, db)

    # 블루프린트
    # 2025-07-25, question_views.py 파일에 등록한 블루프린트 적용을 위한 임포트 (app.register_blueprint() 메서드 사용)
    # 2025-07-25, answer_views.py 파일에 등록한 블루프린트 적용을 위한 임포트
    # 2025-08-11, comment_views.py 파일에 등록한 블루프린트 적용을 위한 임포트
    from .views import main_views, question_views, answer_views, auth_views, comment_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(question_views.bp)
    app.register_blueprint(answer_views.bp)
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(comment_views.bp)

    # 필터
    from .filter import format_datetime
    app.jinja_env.filters['datetime'] = format_datetime

    # 2025-07-29, RAG 챗봇 기능을 위한 블루프린트 등록
    from .rag import bp as rag_chat_bp
    app.register_blueprint(rag_chat_bp)

    # 마크다운 확장 - 현재 주석 처리됨
    # Markdown(app, extensions=['nl2br', 'fenced_code'])

    return app