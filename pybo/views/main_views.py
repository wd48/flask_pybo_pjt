from flask import Blueprint

bp = Blueprint('main', __name__, url_prefix='/')

# 2025-07-23, add a routing function to the main blueprint, 2xy
@bp.route('/hello')
def hello_pybo():
    return 'Hello, Pybo! from main_views.py'

# 함수명이 동일하면 에러나므로 주의
@bp.route('/')
def index():
    return 'Pybo index page'