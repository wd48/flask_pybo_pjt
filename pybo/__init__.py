from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import config
'''
2025-07-25
- db 객체를 create_app 함수 안에서 생성하면 블루프린트와 같은 모듈에서 db 객체를 사용할 수 없으므로
db, migrate 객체를 모듈 레벨에서 생성하고 (create_app 함수 밖), create_app 함수에서 초기화하는 방식으로 변경합니다.
- 해당 객체 (db, migrate)를 앱에 등록할 때는 create_app 함수에서 init_app 함수를 통해 진행한다.
'''

db = SQLAlchemy()
migrate = Migrate()
from . import models  # 모델을 임포트하여 SQLAlchemy가 모델 클래스를 인식하도록 함

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    # ORM
    db.init_app(app)
    migrate.init_app(app, db)

    # 블루프린트
    # 2025-07-25, question_views.py 파일에 등록한 블루프린트 적용을 위한 임포트 (app.register_blueprint() 메서드 사용)
    # 2025-07-25, answer_views.py 파일에 등록한 블루프린트 적용을 위한 임포트
    from .views import main_views, question_views, answer_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(question_views.bp)
    app.register_blueprint(answer_views.bp)

    return app