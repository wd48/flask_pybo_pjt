# pybo/rag/metrics.py
from collections import deque
from datetime import datetime

# 챗봇 응답 시간과 감정 분석 결과를 저장할 deque (최대 200개)
# 실제 환경에서는 데이터베이스 사용을 권장
chatbot_response_times = deque(maxlen=200)

def log_chatbot_response_time(duration: float, source: str):
    """챗봇 응답 시간을 기록합니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chatbot_response_times.append({"timestamp": timestamp, "duration": duration, "source": source})
    print(f"[-METRICS-] Logged response time from '{source}': {duration:.4f}s at {timestamp}")

def get_chatbot_metrics():
    """챗봇 응답 시간 데이터를 반환합니다."""
    return chatbot_response_times