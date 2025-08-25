# pybo/rag/routes.py
import os
import json
from datetime import datetime
from flask import Blueprint, render_template, request, url_for, redirect, flash, jsonify, session, current_app
from langchain.evaluation import load_evaluator
from langchain_community.document_loaders import PyPDFLoader
from .pipeline import ask_rag, run_llm_chain, analyze_sentiment, summarize_text
from .upload_utils import (
    save_pdf_and_index, list_uploaded_pdfs, get_pdf_retriever,
    get_collection_names, get_file_collection_info, delete_collection_and_file
)
from .metrics import get_chatbot_metrics
from .models import get_llm
from .vectorstore import get_persistent_client # get_persistent_client 임포트 추가

import chromadb
from chromadb.api.models.Collection import Collection

bp = Blueprint("rag", __name__, url_prefix="/chat")

# ====== 라우팅 ======
@bp.route("/", methods=["GET"])
def index():
    selected_file = request.args.get('file', default=None, type=str)
    if selected_file is not None:
        session['chat_history'] = []
    
    if 'chat_history' not in session:
        session['chat_history'] = []

    files = list_uploaded_pdfs()
    collection_info = get_file_collection_info()

    return render_template("rag/chat.html",
                           chat_history=session['chat_history'],
                           files=files,
                           collection_info=collection_info,
                           selected_file=selected_file)

# 질문을 처리하는 엔드포인트 (fetch API)
@bp.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    selected_file = request.form.get("filename")
    
    if not question:
        return jsonify({"error": "질문을 입력하세요"}), 400

    session['chat_history'].append({"role": "user", "content": question})

    answer = ""
    if selected_file:
        retriever = get_pdf_retriever(selected_file)
        if retriever:
            answer = run_llm_chain(question, retriever)
        else:
            answer = f"파일 '{selected_file}'의 컬렉션을 찾을 수 없습니다."
    else:
        answer = ask_rag(question)

    # 답변 생성 후 평가 실행 및 로그 저장, 2025-08-22 jylee
    # 평가 기준이 있다면 reference를 제공할 수 있습니다.
    run_and_log_evaluation(question=question, prediction=answer)

    # 세션저장.
    session['chat_history'].append({"role": "bot", "content": answer})
    session.modified = True

    return jsonify({"answer": answer})

# 챗봇 대화 기록을 초기화하는 엔드포인트
@bp.route("/clear", methods=["POST"])
def clear_chat():
    session.pop('chat_history', None)
    return redirect(url_for('rag.index'))

# PDF 파일 요약 엔드포인트
@bp.route("/summarize", methods=["POST"])
def summarize():
    filename = request.form.get("filename")
    if not filename:
        return jsonify({"error": "Filename is required."}),

    try:
        upload_folder = current_app.config["CHAT_UPLOAD_FOLDER"]
        filepath = os.path.join(upload_folder, filename)

        if not os.path.exists(filepath):
            return jsonify({"error": "File not found."}),

        loader = PyPDFLoader(filepath)
        pages = loader.load()
        full_text = "\n".join(page.page_content for page in pages)

        if not full_text.strip():
            return jsonify({"summary": "이 PDF 파일에서 텍스트를 추출할 수 없습니다."})

        summary = summarize_text(full_text)
        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({"error": "요약 생성 중 오류가 발생했습니다."}),

# 파일 관리 페이지 및 업로드 기능
@bp.route("/files", methods=['GET', 'POST'])
def manage_files():
    files = list_uploaded_pdfs()
    collection_info = get_file_collection_info()
    collection_names = get_collection_names()
    
    if request.method == 'POST':
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
            except Exception as e:
                flash(f"파일 업로드 중 오류가 발생했습니다: {str(e)}")
        else:
            flash("PDF 파일만 업로드할 수 있습니다.")
        return redirect(url_for('rag.manage_files'))

    return render_template('rag/manage_files.html',
                           files=files,
                           collection_info=collection_info,
                           collection_names=collection_names)

# 파일 삭제 엔드포인트
@bp.route("/files/delete/<filename>", methods=['POST'])
def delete_file(filename):
    try:
        if delete_collection_and_file(filename):
            flash(f"'{filename}' 파일 및 관련 데이터가 성공적으로 삭제되었습니다.")
        else:
            flash(f"'{filename}' 파일 또는 관련 데이터 삭제에 실패했습니다. 로그를 확인해주세요.")
    except Exception as e:
        flash(f"삭제 중 오류가 발생했습니다: {str(e)}")
    
    return redirect(url_for('rag.manage_files'))

# 설정 페이지 및 컬렉션 관리
@bp.route('/settings')
def settings():
    collection_info = get_file_collection_info() # get_file_collection_info() 사용
    collections_list = []
    for filename, info in collection_info.items():
        collections_list.append({
            "name": info['collection_name'],
            "count": info['document_count']
        })
    print(f"--- Settings Collections: {collections_list} ---")
    return render_template('rag/settings.html', collections=collections_list)

@bp.route('/settings/delete/<collection_name>', methods=['POST'])
def delete_setting_collection(collection_name):
    try:
        persistent_client = get_persistent_client()
        if persistent_client:
            persistent_client.delete_collection(name=collection_name)
            flash(f"컬렉션 '{collection_name}'이(가) 삭제되었습니다.")
    except Exception as e:
        flash(f"컬렉션 삭제 중 오류가 발생했습니다: {str(e)}")
    
    return redirect(url_for('rag.settings'))

# # API 엔드포인트
# @bp.route("/api/collections")
# def api_collections():
#     try:
#         collection_info = get_file_collection_info()
#         collection_names = get_collection_names()
#         
#         return jsonify({
#             'status': 'success',
#             'file_collections': collection_info,
#             'all_collections': collection_names
#         })
#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 500

# 감성 분석 페이지 및 기능, 2025-08-22 jylee
@bp.route("/sentiment_analysis", methods=['GET','POST'])
def sentiment_analysis():
    result = None
    if request.method == "POST":
        # 폼 데이터 가져오기
        gender = request.form.get('gender')
        age = request.form.get('age')
        emotion = request.form.get('emotion')
        meaning = request.form.get('meaning')
        action_list = request.form.getlist('action')
        action = ', '.join(action_list) if action_list else '없음'
        reflect_list = request.form.getlist('reflect')
        reflect = ', '.join(reflect_list) if reflect_list else '없음'
        anchor = request.form.get('anchor')

        # 감성 분석 실행
        result = analyze_sentiment(
            gender=gender, age=age, emotion=emotion, meaning=meaning,
            action=action, reflect=reflect, anchor=anchor
        )

        # 감성 분석 질문 구성
        sentiment_question = f"성별: {gender}, 연령대: {age}, 감정: {emotion}, 이유: {meaning}, 행동: {action}, 성찰: {reflect}, 다짐: {anchor}"
        # 감성 분석 결과 평가 실행 및 로그 저장
        run_and_log_evaluation(question=sentiment_question, prediction=result)

        # 기존 로그 저장 로직
        log_data = {"timestamp": datetime.now().isoformat(), "gender": gender, "age": age, "emotion": emotion, "meaning": meaning, "action": action, "reflect": reflect, "anchor": anchor, "analysis_result": result}
        logs_dir = os.path.join(current_app.root_path, '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = os.path.join(logs_dir, f"sentiment_{timestamp_str}.json")
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
        flash("감정 분석 결과가 기록되었습니다.")

    chatbot_data = get_chatbot_metrics()
    chatbot_labels = [item['timestamp'] for item in chatbot_data]
    chatbot_values = [item['duration'] for item in chatbot_data]

    return render_template(
        "rag/sentiment.html",
        result=result,
        chatbot_labels=chatbot_labels,
        chatbot_values=chatbot_values
    )

# 성능 대시보드, 챗봇 응답 시간 시각화 2025-08-22 jylee
@bp.route("/performance")
def performance_dashboard():
    chatbot_data = get_chatbot_metrics()
    chatbot_labels = [item['timestamp'] for item in chatbot_data]
    chatbot_values = [item['duration'] for item in chatbot_data]

    return render_template(
        "rag/performance.html",
        chatbot_labels=chatbot_labels,
        chatbot_values=chatbot_values
    )


# ====== 평가 공통 함수 ======
# # 평가를 실행하고 결과를 JSON 파일로 저장합니다.
def run_and_log_evaluation(question: str, prediction: str, reference: str = None):
    """답변을 평가하고, 결과를 JSON 파일로 저장한 후, 평가 결과를 반환합니다."""
    try:
        llm = get_llm()

        # 평가기 로드
        relevance_evaluator = load_evaluator("criteria", criteria="relevance", llm=llm)
        conciseness_evaluator = load_evaluator("criteria", criteria="conciseness", llm=llm)

        eval_results = {}

        # 관련성 평가
        relevance_result = relevance_evaluator.evaluate_strings(prediction=prediction, input=question)
        eval_results["relevance"] = relevance_result

        # 간결성 평가
        conciseness_result = conciseness_evaluator.evaluate_strings(prediction=prediction, input=question)
        eval_results["conciseness"] = conciseness_result

        # 정확성 평가 (reference가 있을 경우)
        if reference:
            correctness_evaluator = load_evaluator("qa", llm=llm)
            correctness_result = correctness_evaluator.evaluate_strings(
                prediction=prediction, reference=reference, input=question
            )
            eval_results["correctness"] = correctness_result

        # 로그 데이터 생성
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "prediction": prediction,
            "reference": reference,
            "evaluation": eval_results
        }

        # 로그 디렉토리 및 파일 저장
        logs_dir = os.path.join(current_app.root_path, '..', 'evaluation_logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = os.path.join(logs_dir, f"eval_{timestamp_str}.json")

        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)

        return eval_results

    except Exception as e:
        print(f"--- Evaluation Error ---: {e}")
        # 평가 중 오류가 발생하더라도 메인 기능에 영향을 주지 않도록 처리
        return None


# 평가 결과 페이지 : 저장된 평가 결과 목록을 보여주고, 각 결과를 클릭하면 상세 정보를 보여줍니다.
# 2025-08-22 jylee
@bp.route("/evaluation_results")
def evaluation_results():
    """저장된 평가 결과 목록을 보여줍니다."""
    logs_dir = os.path.join(current_app.root_path, '..', 'evaluation_logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    eval_files = sorted(os.listdir(logs_dir), reverse=True)
    evaluations = []
    for filename in eval_files:
        if filename.endswith(".json"):
            try:
                with open(os.path.join(logs_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['filename'] = filename
                    evaluations.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading or parsing {filename}: {e}")

    return render_template("rag/evaluation_results.html", evaluations=evaluations)