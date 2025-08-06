# pybo/rag_chat/routes.py
from flask import Blueprint, render_template, request, url_for, redirect, flash
from .rag_pipeline import ask_rag
from .upload_utils import save_and_embed_pdf

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

# 2025-08-07 파일 업로드 라우트
@bp.route("/upload", methods=['GET','POST'])
def upload_pdf():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files['pdf_file']
        # 파일이 선택되지 않은 경우
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        if file and file.filename.endswith('.pdf'):
            save_and_embed_pdf(file)
            flash("PDF 파일 업로드 및 임베딩 성공")
            return redirect(url_for('rag_chat.index'))
        else:
            flash("PDF 파일만 업로드할 수 있습니다.")
            return redirect(request.url)

    return render_template("rag_chat/upload.html")