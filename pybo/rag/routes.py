# pybo/rag/routes.py
import json
import os
import threading
import time
from datetime import datetime

from flask import Blueprint, render_template, request, url_for, redirect, flash, jsonify, session, current_app, Response
from langchain.evaluation import load_evaluator
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage

from .metrics import get_chatbot_metrics, log_chatbot_response_time
from .models import get_llm
from .pipeline import summarize_text, get_conversational_rag_chain, analyze_sentiment_stream
from .upload_utils import (
    save_pdf_and_index, list_uploaded_pdfs, get_pdf_retriever,
    get_collection_names, get_file_collection_info, delete_collection_and_file
)
from .vectorstore import get_persistent_client, get_all_file_collections

bp = Blueprint("rag", __name__, url_prefix="/chat")

# 평가기 캐싱을 위한 전역 변수
relevance_evaluator = None
conciseness_evaluator = None
correctness_evaluator = None

# 백그라운드 스레드에서 평가 실행 함수, 2025-09-04 jylee
def run_eval_in_background(app, question: str, prediction: str):
    """백그라운드 스레드에서 평가를 실행합니다."""
    with app.app_context():
        run_and_log_evaluation(question=question, prediction=prediction)

# 백그라운드 스레드 시작 함수, 2025-09-04 jylee
def start_evaluation_in_background(question: str, prediction: str):
    """평가 함수를 백그라운드 스레드에서 실행을 시작합니다."""
    app = current_app._get_current_object()
    eval_thread = threading.Thread(
        target=run_eval_in_background,
        args=(app, question, prediction)
    )
    eval_thread.start()

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
    print(f"--- ask() called with question: '{question}', file: '{selected_file}' ---")
    
    if not question:
        return jsonify({"error": "질문을 입력하세요"}), 400

    # 1. 세션의 chat_history 를 LangChain이 이해하는 형태로 변환, 2025-08-27 jylee
    chat_history_from_session = session.get('chat_history')
    chat_history_for_chain = []
    print(f"--- Chat History from Session: {chat_history_from_session} ---")
    for msg in chat_history_from_session:
        if msg['role'] == 'user':
            chat_history_for_chain.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'bot':
            chat_history_for_chain.append(AIMessage(content=msg['content']))

    # 2. 사용자 질문을 세션에 추가 (UI 표시용)
    session['chat_history'].append({"role": "user", "content": question})
    session.modified = True # 세션이 변경되었음을 플라스크에 알림

    answer = ""
    # 3. Retriever를 가져와 새로운 대화형 RAG 체인 생성
    if selected_file:
        print(f"--- Getting retriever for selected file: {selected_file} ---")
        retriever = get_pdf_retriever(selected_file)
    else:
        print("--- No file selected, getting retriever for all documents ---")
        # 전체 문서 검색 로직 (현재는 첫번째 컬렉션 사용)
        all_collections = get_all_file_collections()
        if not all_collections:
            return jsonify({"error": "사용 가능한 문서 컬렉션이 없습니다."} ), 500
        first_collection_key = next(iter(all_collections))
        retriever = all_collections[first_collection_key]['vectordb'].as_retriever()
    if not retriever:
        print("--- Failed to get retriever ---")
        return jsonify({"answer": "문서 검색기를 준비할 수 없습니다."} )

    conversational_rag_chain = get_conversational_rag_chain(retriever)

    # 4. 변환된 대화 기록과 새 질문으로 체인 실행
    print("--- Invoking conversational RAG chain ---")
    start_time = time.time()
    result = conversational_rag_chain.invoke(
        {"input": question,"chat_history": chat_history_for_chain}
    )
    end_time = time.time()
    answer = result["answer"]
    log_chatbot_response_time(end_time - start_time, source="챗봇")
    print(f"--- RAG chain result: {result} ---")

    # 답변 생성 후 평가를 백그라운드에서 실행하여 응답 지연 방지, 2025-09-04 jylee
    start_evaluation_in_background(question, answer)

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
    print(f"--- summarize() called for file: {filename} ---")
    if not filename:
        return jsonify({"error": "Filename is required."} ), 400

    try:
        upload_folder = current_app.config["CHAT_UPLOAD_FOLDER"]
        filepath = os.path.join(upload_folder, filename)

        if not os.path.exists(filepath):
            print(f"--- File not found at path: {filepath} ---")
            return jsonify({"error": "File not found."} ), 404

        loader = PyPDFLoader(filepath)
        pages = loader.load()
        full_text = "\n".join(page.page_content for page in pages)

        if not full_text.strip():
            print(f"--- No text could be extracted from PDF: {filename} ---")
            return jsonify({"summary": "이 PDF 파일에서 텍스트를 추출할 수 없습니다."} )

        summary = summarize_text(full_text)
        print(f"--- Summary generated for {filename} ---")
        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({"error": "요약 생성 중 오류가 발생했습니다."} ), 500

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
                print(f"--- Uploading and indexing file: {file.filename} ---")
                chunk_count = save_pdf_and_index(file)
                print(f"--- File '{file.filename}' uploaded successfully with {chunk_count} chunks ---")
                flash(f"PDF 파일이 성공적으로 업로드되었습니다. ({chunk_count}개 청크 생성)")
            except Exception as e:
                print(f"--- Error uploading file '{file.filename}': {e} ---")
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
    print(f"--- delete_file() called for file: {filename} ---")
    try:
        if delete_collection_and_file(filename):
            print(f"--- Successfully deleted file and collection for: {filename} ---")
            flash(f"'{filename}' 파일 및 관련 데이터가 성공적으로 삭제되었습니다.")
        else:
            print(f"--- Failed to delete file or collection for: {filename} ---")
            flash(f"'{filename}' 파일 또는 관련 데이터 삭제에 실패했습니다. 로그를 확인해주세요.")
    except Exception as e:
        print(f"--- Error deleting file '{filename}': {e} ---")
        flash(f"삭제 중 오류가 발생했습니다: {str(e)}")
    
    return redirect(url_for('rag.manage_files'))

# 설정 페이지 및 컬렉션 관리
@bp.route('/settings')
def settings():
    # 파일과 연결된 컬렉션 정보 가져오기
    collection_info = get_file_collection_info()
    collections_list = []
    for filename, info in collection_info.items():
        collections_list.append({
            "name": info['collection_name'],
            "count": info['document_count'],
            "filename": filename  # 원본 파일명도 함께 전달
        })
    
    # 전체 컬렉션 이름 가져오기
    all_collection_names = get_collection_names()

    print(f"--- Settings Collections: {collections_list} ---")
    return render_template('rag/settings.html', 
                           collections=collections_list, 
                           all_collection_names=all_collection_names)

@bp.route('/settings/delete/<collection_name>', methods=['POST'])
def delete_setting_collection(collection_name):
    print(f"--- delete_setting_collection() called for collection: {collection_name} ---")
    try:
        persistent_client = get_persistent_client()
        if persistent_client:
            persistent_client.delete_collection(name=collection_name)
            print(f"--- Collection '{collection_name}' deleted successfully ---")
            flash(f"컬렉션 '{collection_name}'이(가) 삭제되었습니다.")
    except Exception as e:
        print(f"--- Error deleting collection '{collection_name}': {e} ---")
        flash(f"컬렉션 삭제 중 오류가 발생했습니다: {str(e)}")
    
    return redirect(url_for('rag.settings'))

# 감성 분석 페이지 및 기능, 2025-08-22 jylee
@bp.route("/sentiment_analysis", methods=['GET'])
def sentiment_analysis():
    # 쿼리 파라미터에 'gender'가 있는지 확인하여 일반 페이지 로드와 스트리밍 요청을 구분합니다.
    if 'gender' in request.args:
        # 스트리밍 요청 처리
        gender = request.args.get('gender')
        age = request.args.get('age')
        emotion = request.args.get('emotion')
        meaning = request.args.get('meaning')
        action_list = request.args.getlist('action')
        action = ', '.join(action_list) if action_list else '없음'
        reflect_list = request.args.getlist('reflect')
        reflect = ', '.join(reflect_list) if reflect_list else '없음'
        anchor = request.args.get('anchor')

        # 스트리밍 함수에 전달할 설정값 (컨텍스트가 활성 상태일 때 미리 복사), Application Context 에러 방지
        config = current_app.config.copy()

        def generate_stream():
            stream = analyze_sentiment_stream(
                config=config,
                gender=gender, age=age, emotion=emotion, meaning=meaning,
                action=action, reflect=reflect, anchor=anchor
            )
            for chunk in stream:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "event: end\ndata: {}\n\n"

        return Response(generate_stream(), mimetype='text/event-stream')

    # 일반 GET 요청 시에는 차트 데이터 없이 기본 페이지만 렌더링합니다.
    return render_template("rag/sentiment.html")

# 감정 분석 결과 로깅 엔드포인트, 2025-08-29 jylee
# 스트리밍 방식으로 변경되어 별도 함수로 구현
@bp.route("/log_sentiment_result", methods=['POST'])
def log_sentiment_result():
    data = request.json
    try:
        # 원본 질문 구성
        sentiment_question = f"성별: {data['gender']}, 연령대: {data['age']}, 감정: {data['emotion']}, 이유: {data['meaning']}, 행동: {data['action']}, 성찰: {data['reflect']}, 다짐: {data['anchor']}"
        
        # 평가 및 로그 실행 (백그라운드에서 실행), 2025-09-04 jylee
        start_evaluation_in_background(sentiment_question, data['result'])

        # 기존 로그 저장 로직
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "gender": data['gender'],
            "age": data['age'],
            "emotion": data['emotion'],
            "meaning": data['meaning'],
            "action": data['action'],
            "reflect": data['reflect'],
            "anchor": data['anchor'],
            "analysis_result": data['result']
        }
        logs_dir = os.path.join(current_app.root_path, '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = os.path.join(logs_dir, f"sentiment_{timestamp_str}.json")
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=4)
        
        return jsonify({"status": "success", "message": "로그가 성공적으로 저장되었습니다."} ), 200
    except Exception as e:
        print(f"--- Error during sentiment logging: {e} ---")
        return jsonify({"status": "error", "message": str(e)}), 500

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
# 평가를 실행하고 결과를 JSON 파일로 저장합니다.
def run_and_log_evaluation(question: str, prediction: str, reference: str = None):
    """답변을 평가하고, 결과를 JSON 파일로 저장한 후, 평가 결과를 반환합니다."""
    global relevance_evaluator, conciseness_evaluator, correctness_evaluator
    print(f"--- Starting evaluation for question: '{question[:50]}...' ---")
    try:
        llm = get_llm()

        # 평가기 캐싱: 최초 호출 시에만 로드
        if relevance_evaluator is None:
            print("--- Initializing relevance evaluator ---")
            relevance_evaluator = load_evaluator("criteria", criteria="relevance", llm=llm)
        if conciseness_evaluator is None:
            print("--- Initializing conciseness evaluator ---")
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
            if correctness_evaluator is None:
                print("--- Initializing correctness evaluator ---")
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

        print(f"--- Evaluation successful. Log saved to {log_file_path} ---")
        return eval_results

    except Exception as e:
        print(f"--- Evaluation Error ---: {e}")
        # 평가 중 오류가 발생하더라도 메인 기능에 영향을 주지 않도록 처리
        return None


# 평가 결과 페이지 : 저장된 평가 결과 목록을 보여주고, 각 결과를 클릭하면 상세 정보를 보여줍니다.
# 2025-08-22 jylee
@bp.route("/evaluation_results")
def evaluation_results():
    """저장된 평가 결과 목록과 성능 차트를 보여줍니다."""
    # 1. 평가 결과 로드
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

    # 2. 성능 차트 데이터 처리, 2025-08-29 jylee
    chatbot_data = get_chatbot_metrics()
    
    # 데이터를 source별로 분리하고 타임스탬프를 기준으로 정렬
    all_timestamps = sorted(list(set(item['timestamp'] for item in chatbot_data)))
    
    chatbot_metrics = {}
    sentiment_metrics = {}
    for item in chatbot_data:
        if item['source'] == '챗봇':
            chatbot_metrics[item['timestamp']] = item['duration']
        elif item['source'] == '감정 분석':
            sentiment_metrics[item['timestamp']] = item['duration']
            
    # Chart.js 데이터셋 형식에 맞춤 (해당 타임스탬프에 데이터가 없으면 null 처리)
    chatbot_values = [chatbot_metrics.get(ts) for ts in all_timestamps]
    sentiment_values = [sentiment_metrics.get(ts) for ts in all_timestamps]

    return render_template(
        "rag/evaluation_results.html", 
        evaluations=evaluations,
        chart_labels=all_timestamps,
        chatbot_chart_values=chatbot_values,
        sentiment_chart_values=sentiment_values
    )
