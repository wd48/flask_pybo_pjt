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
    get_collection_names, get_file_collection_info, delete_collection_and_file,
    save_kb_and_index, list_uploaded_kbs, delete_kb_collection_and_file
)
from .vectorstore import get_persistent_client, get_all_file_collections

bp = Blueprint("rag", __name__, url_prefix="/chat")

# 평가기 캐싱을 위한 전역 변수
relevance_evaluator = None
conciseness_evaluator = None
correctness_evaluator = None

# 백그라운드 스레드에서 평가 실행 함수 (로그 경로 통일, 2025-09-04 jylee)
def run_eval_in_background(app, question: str, prediction: str, log_type: str = 'normal_rag', full_data: dict = None):
    """백그라운드 스레드에서 평가 및 로그 저장을 실행합니다."""
    with app.app_context():
        run_and_log_evaluation(question=question, prediction=prediction, log_type=log_type, full_data=full_data)

# 백그라운드 스레드 시작 함수 (로그 경로 통일, 2025-09-15 jylee)
def start_evaluation_in_background(question: str, prediction: str, log_type: str = 'normal_rag', full_data: dict = None):
    """평가 함수를 백그라운드 스레드에서 실행을 시작합니다."""
    app = current_app._get_current_object()
    eval_thread = threading.Thread(
        target=run_eval_in_background,
        args=(app, question, prediction, log_type, full_data)
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
    chat_history_from_session = session.get('chat_history', [])
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
    # 4. RAG 체인 생성 및 질문 처리
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

    # normal_rag 타입으로 백그라운드 평가 실행
    start_evaluation_in_background(question, answer, log_type='normal_rag')

    # 세션저장.
    session['chat_history'].append({"role": "bot", "content": answer})
    session.modified = True

    return jsonify({"answer": answer})

# 감정 분석 결과 로깅 엔드포인트, 2025-09-15 jylee
@bp.route("/log_sentiment_result", methods=['POST'])
def log_sentiment_result():
    data = request.json
    try:
        sentiment_question = f"성별: {data['gender']}, 연령대: {data['age']}, 감정: {data['emotion']}, 이유: {data['meaning']}, 행동: {data['action']}, 성찰: {data['reflect']}, 다짐: {data['anchor']}"
        
        # sentiment_rag 타입으로 백그라운드 평가 및 로깅 실행
        start_evaluation_in_background(
            question=sentiment_question, 
            prediction=data['result'],
            log_type='sentiment_rag',
            full_data=data
        )
        
        return jsonify({"status": "success", "message": "로그가 성공적으로 저장되었습니다."} ), 200
    except Exception as e:
        print(f"--- Error during sentiment logging: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ====== 평가 공통 함수 ======
def run_and_log_evaluation(question: str, prediction: str, reference: str = None, log_type: str = 'normal_rag', full_data: dict = None):
    """답변을 평가하고, 결과를 지정된 로그 경로에 JSON 파일로 저장합니다."""
    global relevance_evaluator, conciseness_evaluator, correctness_evaluator
    print(f"--- Starting evaluation for question: '{question[:50]}...' (Log type: {log_type}) ---")
    try:
        llm = get_llm()

        if relevance_evaluator is None: relevance_evaluator = load_evaluator("criteria", criteria="relevance", llm=llm)
        if conciseness_evaluator is None: conciseness_evaluator = load_evaluator("criteria", criteria="conciseness", llm=llm)

        eval_results = {}
        eval_results["relevance"] = relevance_evaluator.evaluate_strings(prediction=prediction, input=question)
        eval_results["conciseness"] = conciseness_evaluator.evaluate_strings(prediction=prediction, input=question)

        if reference:
            if correctness_evaluator is None: correctness_evaluator = load_evaluator("qa", llm=llm)
            eval_results["correctness"] = correctness_evaluator.evaluate_strings(prediction=prediction, reference=reference, input=question)

        # 로그 데이터 생성
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "prediction": prediction,
            "reference": reference,
            "evaluation": eval_results
        }

        # sentiment_rag 타입일 경우, 전체 데이터 추가
        if log_type == 'sentiment_rag' and full_data:
            log_data.update(full_data)

        # 로그 타입에 따라 저장 경로 결정 (기본값: normal_rag)
        if log_type == 'sentiment_rag':
            log_folder = 'sentiment_rag'
        else:
            log_folder = 'normal_rag'
        logs_dir = os.path.join(current_app.root_path, '..', 'logs', log_folder)
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = os.path.join(logs_dir, f"log_{timestamp_str}.json")

        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)

        print(f"--- Evaluation successful. Log saved to {log_file_path} ---")
        return eval_results

    except Exception as e:
        print(f"--- Evaluation Error ---: {e}")
        return None

# 챗봇 대화 기록을 초기화하는 엔드포인트
@bp.route("/clear", methods=["POST"])
def clear_chat():
    session.pop('chat_history', None)
    return redirect(url_for('rag.index'))

# PDF 파일 요약 엔드포인트
@bp.route("/summarize", methods=["POST"])
def summarize():
    filename = request.form.get("filename")
    # 파일 이름이 제공되지 않은 경우 오류 반환
    if not filename:
        return jsonify({"error": "Filename is required."} ), 400

    try:
        upload_folder = current_app.config["CHAT_UPLOAD_FOLDER"]
        filepath = os.path.join(upload_folder, filename)
        # 파일이 존재하지 않는 경우 오류 반환
        if not os.path.exists(filepath):
            print(f"--- File not found at path: {filepath} ---")
            return jsonify({"error": "File not found."} ), 404

        loader = PyPDFLoader(filepath)
        pages = loader.load()
        full_text = "\n".join(page.page_content for page in pages)
        # 추출된 텍스트가 없는 경우 오류 반환
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
    # PDF 파일 업로드 처리
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

    # GET 요청 처리
    files = list_uploaded_pdfs()
    collection_info = get_file_collection_info()
    
    # 파일과 연결된 컬렉션 정보 가공
    collections_list = []
    for filename, info in collection_info.items():
        collections_list.append({
            "name": info['collection_name'],
            "count": info['document_count'],
            "filename": filename
        })

    # 전체 컬렉션 이름 가져오기
    all_collection_names = get_collection_names()
    
    # 파일과 연결되지 않은 컬렉션 찾기
    file_collection_names = {info['collection_name'] for info in collection_info.values()}
    unlinked_collections = [name for name in all_collection_names if name not in file_collection_names]

    return render_template('rag/manage_files.html', 
                           files=files, 
                           collection_info=collection_info, 
                           collections=collections_list,
                           all_collection_names=all_collection_names,
                           unlinked_collections=unlinked_collections)

# 지식 베이스 관리 페이지
@bp.route("/kb", methods=['GET', 'POST'])
def manage_kb():
    if request.method == 'POST':
        if 'kb_file' not in request.files:
            flash("선택된 파일이 없습니다.")
            return redirect(request.url)
        file = request.files['kb_file']
        if file.filename == '':
            flash("선택된 파일이 없습니다.")
            return redirect(request.url)
        
        if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            try:
                chunk_count = save_kb_and_index(file)
                flash(f"지식 베이스 파일 '{file.filename}'이 성공적으로 업로드 및 인덱싱되었습니다. ({chunk_count}개 청크)")
            except Exception as e:
                flash(f"지식 베이스 파일 업로드 중 오류 발생: {e}")
        else:
            flash("PDF 또는 TXT 파일만 업로드할 수 있습니다.")
        return redirect(url_for('rag.manage_kb'))

    # GET 요청
    kb_files = list_uploaded_kbs()
    return render_template('rag/manage_kb.html', files=kb_files)

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

# 컬렉션 삭제 엔드포인트
@bp.route('/files/delete_collection/<collection_name>', methods=['POST'])
def delete_collection(collection_name):
    print(f"--- delete_collection() called for collection: {collection_name} ---")
    try:
        persistent_client = get_persistent_client()
        if persistent_client:
            persistent_client.delete_collection(name=collection_name)
            print(f"--- Collection '{collection_name}' deleted successfully ---")
            flash(f"컬렉션 '{collection_name}'이(가) 삭제되었습니다.")
    except Exception as e:
        print(f"--- Error deleting collection '{collection_name}': {e} ---")
        flash(f"컬렉션 삭제 중 오류가 발생했습니다: {str(e)}")
    return redirect(url_for('rag.manage_files'))

# 지식 베이스 파일 삭제 엔드포인트, 2025-09-12 jylee
@bp.route("/kb/delete/<filename>", methods=['POST'])
def delete_kb_file(filename):
    try:
        if delete_kb_collection_and_file(filename):
            flash(f"지식 베이스 파일 '{filename}'이 성공적으로 삭제되었습니다.")
        else:
            flash(f"지식 베이스 파일 '{filename}' 삭제에 실패했습니다.")
    except Exception as e:
        flash(f"지식 베이스 파일 삭제 중 오류 발생: {e}")
    return redirect(url_for('rag.manage_kb'))

# 감성 분석 페이지 및 기능
@bp.route("/sentiment_analysis", methods=['GET'])
def sentiment_analysis():
    # 쿼리 파라미터에 'gender'가 있는지 확인하여 일반 페이지 로드와 스트리밍 요청을 구분합니다.
    if 'gender' in request.args:
        # 스트리밍 요청 처리
        gender = request.args.get('gender')
        age = request.args.get('age')
        emotion = request.args.get('emotion')
        meaning = request.args.get('meaning')
        action = ', '.join(request.args.getlist('action')) if request.args.getlist('action') else '없음'
        reflect = ', '.join(request.args.getlist('reflect')) if request.args.getlist('reflect') else '없음'
        anchor = request.args.get('anchor')

        # 스트리밍 함수에 전달할 설정값 (컨텍스트가 활성 상태일 때 미리 복사), Application Context 에러 방지
        config = current_app.config.copy()
        def generate_stream():
            # RAG 기반 스트리밍 함수는 전체 입력을 하나의 딕셔너리로 받음
            input_data = {
                "gender": gender, "age": age, "emotion": emotion, "meaning": meaning,
                "action": action, "reflect": reflect, "anchor": anchor
            }
            stream = analyze_sentiment_stream(config=config, **input_data)
            for chunk in stream:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "event: end\ndata: {}\n\n"
        return Response(generate_stream(), mimetype='text/event-stream')

    # 일반 GET 요청 시에는 차트 데이터 없이 기본 페이지만 렌더링합니다.
    return render_template("rag/sentiment.html")

# 성능 대시보드
@bp.route("/performance")
def performance_dashboard():
    chatbot_data = get_chatbot_metrics()
    chatbot_labels = [item['timestamp'] for item in chatbot_data]
    chatbot_values = [item['duration'] for item in chatbot_data]
    return render_template("rag/performance.html", chatbot_labels=chatbot_labels, chatbot_values=chatbot_values)

# 평가 결과 페이지
@bp.route("/evaluation_results")
def evaluation_results():
    # 이 라우트는 이제 단순한 파일 목록 표시 기능만 필요
    # 실제 로그는 logs/normal_rag 및 logs/sentiment_rag 에 저장됨
    log_folders = ['normal_rag', 'sentiment_rag']
    all_evaluations = []
    base_logs_dir = os.path.join(current_app.root_path, '..', 'logs')
    # 각 로그 폴더에서 JSON 파일 읽기
    for folder in log_folders:
        folder_path = os.path.join(base_logs_dir, folder)
        if not os.path.exists(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path), reverse=True):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['filename'] = os.path.join(folder, filename)
                        data['log_type'] = folder
                        all_evaluations.append(data)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading or parsing {filename}: {e}")

    # 성능 차트 데이터 처리
    chatbot_data = get_chatbot_metrics()
    
    # 데이터를 source별로 분리하고 타임스탬프를 기준으로 정렬
    all_timestamps = sorted(list(set(item['timestamp'] for item in chatbot_data)))
    chatbot_metrics = {item['timestamp']: item['duration'] for item in chatbot_data if item['source'] == '챗봇'}
    sentiment_metrics = {item['timestamp']: item['duration'] for item in chatbot_data if item['source'] == '감정 분석'}
    chatbot_values = [chatbot_metrics.get(ts) for ts in all_timestamps]
    sentiment_values = [sentiment_metrics.get(ts) for ts in all_timestamps]

    normal_evals = [e for e in all_evaluations if e['log_type'] == 'normal_rag']
    sentiment_evals = [e for e in all_evaluations if e['log_type'] == 'sentiment_rag']

    return render_template(
        "rag/evaluation_results.html", 
        normal_evals=normal_evals,
        sentiment_evals=sentiment_evals,
        chart_labels=all_timestamps,
        chatbot_chart_values=chatbot_values,
        sentiment_chart_values=sentiment_values
    )