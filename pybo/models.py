from pybo import db
'''
2025-07-25
db.Model 클래스를 상속하여 만든다
- 이 때 사용한 db 객체는 pybo/__init__.py에서 생성한 SQLAlchemy 클래스의 객체이다.
- db.Model 클래스를 상속받는 이유는 SQLAlchemy가 제공하는 ORM 기능을 사용하기 위함이다.
'''

# 추천인 : 질문에 투표하는 사용자와 질문 간의 다대다 관계를 정의
question_voter = db.Table(
    'question_voter',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), primary_key=True),
    db.Column('question_id', db.Integer, db.ForeignKey('question.id', ondelete='CASCADE'), primary_key=True)
)

# 추천인 : 답변에 투표하는 사용자와 답변 간의 다대다 관계를 정의
answer_voter = db.Table(
    'answer_voter',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), primary_key=True),
    db.Column('answer_id', db.Integer, db.ForeignKey('answer.id', ondelete='CASCADE'), primary_key=True)
)

# 질문 모델 생성
class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text(), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    user = db.relationship('User', backref=db.backref('question_set'))
    modify_date = db.Column(db.DateTime(), nullable=True)
    voter = db.relationship('User', secondary=question_voter, backref=db.backref('question_voter_set'))

# 답변 모델 생성
'''
question_id : 답변이 속한 질문의 ID를 저장하는 외래키(ForeignKey)
- 답변을 질문과 연결하기 위해 추가한 속성
question.id : question 테이블의 id 컬럼을 참조하는 외래키(ForeignKey)
ondelete='CASCADE' : 질문이 삭제되면 해당 질문에 속한 답변도 함께 삭제되도록 설정
(CASCADE : 데이터베이스 설정으로, 질문을 DB 툴에서 쿼리로 삭제할 때만 질문에 속한 답변들이 삭제된다.)

question : 답변이 속한 질문을 참조하는 관계 설정
- db.relationship()
: Question 모델과의 관계를 정의, 답변 모델에서 연결된 질문 모델의 제목을 answer.question.subject와 같이 접근할 수 있다.

- backref 
: Question 모델에서 answer_set 속성을 통해 해당 질문에 속한 답변들을 조회할 수 있도록 설정
'''
class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id', ondelete='CASCADE'))
    question = db.relationship('Question', backref=db.backref('answer_set'))
    content = db.Column(db.Text(), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    user = db.relationship('User', backref=db.backref('answer_set'))
    modify_date = db.Column(db.DateTime(), nullable=True)
    voter = db.relationship('User', secondary=answer_voter, backref=db.backref('answer_voter_set'))

# 회원가입
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)