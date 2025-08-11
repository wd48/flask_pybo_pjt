# pybo/rag_chat/routes.py
from flask import Blueprint, render_template, request, url_for, redirect, flash
from .rag_pipeline import ask_rag, run_llm_chain, analyze_sentiment
from .upload_utils import save_and_embed_pdf, list_uploaded_pdfs, query_by_pdf

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

# 2025-08-07 PDF 파일 목록 라우트
@bp.route("/files", methods=['GET'])
def file_list():
    files = list_uploaded_pdfs()
    return render_template("rag_chat/file_list.html", files=files)

# 2025-08-07 PDF 파일 검색 라우트
@bp.route('/search_by_file', methods=['POST','GET'])
def search_by_file():
    filename = request.form.get('filename')
    query = request.form.get('query')

    if not filename or not query:
        flash("파일과 질문을 모두 입력하세요.")
        return redirect(url_for('rag_chat.file_list'))

    retriever = query_by_pdf(filename, query)
    answer = run_llm_chain(query, retriever)
    return render_template('rag_chat/file_list.html', files=list_uploaded_pdfs(), answer=answer, selected_file=filename)

# 2025-08-11 감정분석
@bp.route("/sentiment", methods=['GET','POST'])
def sentiment():
    result = None
    if request.method == "POST":
        text = request.form.get("text")
        result = analyze_sentiment(text)
    return render_template("rag_chat/sentiment.html", result=result)