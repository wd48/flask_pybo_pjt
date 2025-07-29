# pybo/rag_chat/routes.py
from flask import Blueprint, render_template, request
from .rag_pipeline import ask_rag

bp = Blueprint("rag_chat", __name__, url_prefix="/chat")

# ====== 라우팅 ======
@bp.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            answer = ask_rag(question)
    return render_template("rag_chat/chat.html", answer=answer)