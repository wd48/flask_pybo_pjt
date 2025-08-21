# pybo/rag/metrics.py
from collections import defaultdict
import time
from datetime import datetime

# 챗봇 응답 시간과 감정 분석 결과를 저장할 인메모리 리스트
# 실제 환경에서는 데이터베이스 사용을 권장
chatbot_response_times = []
sentiment_results = defaultdict(int)

def log_chatbot_response_time(duration: float):
    """챗봇 응답 시간을 기록합니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chatbot_response_times.append({"timestamp": timestamp, "duration": duration})
    print(f"[-METRICS-] Logged chatbot response time: {duration:.4f}s at {timestamp}")

def log_sentiment_result(sentiment_class: str):
    """감정 분석 결과를 기록합니다."""
    sentiment_results[sentiment_class] += 1
    print(f"[-METRICS-] Logged sentiment result: {sentiment_class}")

def get_chatbot_metrics():
    """챗봇 응답 시간 데이터를 반환합니다."""
    return chatbot_response_times

def get_sentiment_metrics():
    """감정 분석 결과 분포를 반환합니다."""
    return sentiment_results