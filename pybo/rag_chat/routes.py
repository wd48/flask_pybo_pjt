# pybo/rag_chat/routes.py
from flask import Blueprint, render_template, request, url_for, redirect, flash
from .pipeline import ask_rag, run_llm_chain, analyze_sentiment
from .upload_utils import save_pdf_and_index, list_uploaded_pdfs, query_by_pdf, get_pdf_retriever, get_collection_names

bp = Blueprint("rag_chat", __name__, url_prefix="/chat")

# ====== 라우팅 ======
@bp.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    selected_file = ""
    files = list_uploaded_pdfs()
    
    if request.method == "POST":
        question = request.form.get("question")
        selected_file = request.form.get("filename")
        if question and selected_file:
            retriever = get_pdf_retriever(selected_file)
            answer = run_llm_chain(question, retriever)
            print(f"[-RAG-] Answer generated for question: {question} using file: {selected_file}")
        elif question:
            answer = ask_rag(question)
            print(f"[-RAG-] [no file] Answer generated for question: {question} without file")
        else:
            flash("파일과 질문을 모두 입력하세요")

    return render_template("rag_chat/chat.html", answer=answer, files=files, selected_file=selected_file)

# 2025-08-07 PDF 파일 목록 라우트
# 2025-08-13 파일 업로드 및 검색 기능 통합
# 파일 업로드 및 검색 기능을 제공하는 라우트
@bp.route("/files", methods=['GET', 'POST'])
def manage_files():
    files = list_uploaded_pdfs()

    collection_names = get_collection_names()
    print(f"[-RAG-] Available collections: {collection_names}")

    if request.method == 'POST':
        # 파일 업로드 요청 처리
        if 'pdf_file' not in request.files:
            flash("선택된 파일이 없습니다.")
            return redirect(request.url)

        file = request.files['pdf_file']
        if file.filename == '':
            flash("선택된 파일이 없습니다.")
            return redirect(request.url)

        if file and file.filename.endswith('.pdf'):
            save_pdf_and_index(file)
            flash("PDF 파일이 성공적으로 업로드되었습니다.")
            return redirect(url_for('rag_chat.manage_files'))
        else:
            flash("PDF 파일만 업로드할 수 있습니다.")
            return redirect(url_for('rag_chat.manage_files'))

    return render_template('rag_chat/manage_files.html', files=files)

# 2025-08-11 감정분석
@bp.route("/sentiment", methods=['GET','POST'])
def sentiment():
    result = None
    if request.method == "POST":
        text = request.form.get("text")
        result = analyze_sentiment(text)
    return render_template("rag_chat/sentiment.html", result=result)