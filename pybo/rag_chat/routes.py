# pybo/rag_chat/routes.py
import chromadb
from flask import Blueprint, render_template, request, url_for, redirect, flash, jsonify

from config import CHAT_DB_PERSIST_DIR
from .pipeline import ask_rag, run_llm_chain, analyze_sentiment
from .upload_utils import (
    save_pdf_and_index, list_uploaded_pdfs, get_pdf_retriever, 
    get_collection_names, get_file_collection_info, delete_collection_and_file
)
from .metrics import get_chatbot_metrics, get_sentiment_metrics

bp = Blueprint("rag_chat", __name__, url_prefix="/chat")

# ====== 라우팅 ======
@bp.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    selected_file = ""
    files = list_uploaded_pdfs()
    collection_info = get_file_collection_info()

    if request.method == "POST":
        question = request.form.get("question")
        selected_file = request.form.get("filename")
        if question and selected_file:
            retriever = get_pdf_retriever(selected_file)
            if retriever:
                answer = run_llm_chain(question, retriever)
                print(f"[-RAG-] Answer generated for question: {question} using file: {selected_file}")
            else:
                flash(f"파일 '{selected_file}'의 컬렉션을 찾을 수 없습니다.")
        elif question:
            answer = ask_rag(question)
            print(f"[-RAG-] [no file] Answer generated for question: {question} without file")
        else:
            flash("파일과 질문을 모두 입력하세요")

    return render_template("rag_chat/chat.html", 
                           answer=answer, 
                           files=files, 
                           selected_file=selected_file,
                           collection_info=collection_info)

# 2025-08-07 PDF 파일 목록 라우트
# 2025-08-13 파일 업로드 및 검색 기능 통합
# 파일 업로드 및 검색 기능을 제공하는 라우트
@bp.route("/files", methods=['GET', 'POST'])
def manage_files():
    """파일 업로드 및 컬렉션 관리 페이지"""
    files = list_uploaded_pdfs()
    collection_info = get_file_collection_info()
    collection_names = get_collection_names()
    
    print(f"[-RAG-] Available collections: {collection_names}")
    print(f"[-RAG-] File collection info: {collection_info}")

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
            try:
                chunk_count = save_pdf_and_index(file)
                flash(f"PDF 파일이 성공적으로 업로드되었습니다. ({chunk_count}개 청크 생성)")
                return redirect(url_for('rag_chat.manage_files'))
            except Exception as e:
                flash(f"파일 업로드 중 오류가 발생했습니다: {str(e)}")
                return redirect(url_for('rag_chat.manage_files'))
        else:
            flash("PDF 파일만 업로드할 수 있습니다.")
            return redirect(url_for('rag_chat.manage_files'))

    return render_template('rag_chat/manage_files.html', 
                         files=files, 
                         collection_info=collection_info,
                         collection_names=collection_names)

# 파일 컬렉션 삭제 라우트
@bp.route("/files/delete/<filename>", methods=['POST'])
def delete_file(filename):
    """특정 파일의 컬렉션과 물리적 파일을 삭제합니다."""
    try:
        if delete_collection_and_file(filename):
            flash(f"'{filename}' 파일 및 관련 데이터가 성공적으로 삭제되었습니다.")
        else:
            flash(f"'{filename}' 파일 또는 관련 데이터 삭제에 실패했습니다. 로그를 확인해주세요.")
    except Exception as e:
        flash(f"삭제 중 오류가 발생했습니다: {str(e)}")
    
    return redirect(url_for('rag_chat.manage_files'))

# DB 설정 페이지
@bp.route('/settings')
def settings():
    """DB 관리 페이지"""
    collections_info = []
    try:
        persistent_client = chromadb.PersistentClient(CHAT_DB_PERSIST_DIR)
        collections = persistent_client.list_collections()
        for collection in collections:
            collections_info.append({
                "name": collection.name,
                "count": collection.count()
            })
    except Exception as e:
        flash(f"컬렉션 정보를 가져오는 중 오류가 발생했습니다: {e}")
    
    return render_template('rag_chat/settings.html', collections=collections_info)

@bp.route('/settings/delete/<collection_name>', methods=['POST'])
def delete_setting_collection(collection_name):
    """DB 관리 페이지에서 컬렉션을 삭제합니다."""
    try:
        persistent_client = chromadb.PersistentClient(CHAT_DB_PERSIST_DIR)
        persistent_client.delete_collection(name=collection_name)
        flash(f"컬렉션 '{collection_name}'이(가) 삭제되었습니다.")
    except Exception as e:
        flash(f"컬렉션 삭제 중 오류가 발생했습니다: {str(e)}")
    
    return redirect(url_for('rag_chat.settings'))


# 컬렉션 정보 API
@bp.route("/api/collections")
def api_collections():
    """컬렉션 정보를 JSON으로 반환합니다."""
    try:
        collection_info = get_file_collection_info()
        collection_names = get_collection_names()
        
        return jsonify({
            'status': 'success',
            'file_collections': collection_info,
            'all_collections': collection_names
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# 2025-08-11 감정분석
@bp.route("/sentiment", methods=['GET','POST'])
def sentiment_analysis():
    """감정 분석 페이지"""
    result = None
    if request.method == "POST":
        gender = request.form.get('gender')
        age = request.form.get('age')
        emotion = request.form.get('emotion')
        meaning = request.form.get('meaning')
        action_list = request.form.getlist('action')  # 체크박스는 리스트로 받아옴
        action = ', '.join(action_list) if action_list else '없음'
        reflect_list = request.form.getlist('reflect')  # 체크박스는 리스트로 받아옴
        reflect = ', '.join(reflect_list) if reflect_list else '없음'
        anchor = request.form.get('anchor')

        result = analyze_sentiment(
            gender=gender,
            age=age,
            emotion=emotion,
            meaning=meaning,
            action=action,
            reflect=reflect,
            anchor=anchor
        )

    return render_template("rag_chat/sentiment.html", result=result)


# 2025-08-18 성능 시각화 페이지 라우트
@bp.route("/performance")
def performance_dashboard():
    """성능 대시보드 페이지"""
    chatbot_data = get_chatbot_metrics()
    sentiment_data = get_sentiment_metrics()

    # Chart.js에 전달할 데이터 형식으로 변환
    chatbot_labels = [item['timestamp'] for item in chatbot_data]
    chatbot_values = [item['duration'] for item in chatbot_data]

    sentiment_labels = list(sentiment_data.keys())
    sentiment_values = list(sentiment_data.values())

    return render_template(
        "rag_chat/performance.html",
        chatbot_labels=chatbot_labels,
        chatbot_values=chatbot_values,
        sentiment_labels=sentiment_labels,
        sentiment_values=sentiment_values
    )