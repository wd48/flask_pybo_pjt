from huggingface_hub import snapshot_download
import os

# 다운로드할 모델 이름
model_name = "jhgan/ko-sroberta-multitask"
# 모델을 저장할 로컬 디렉터리
local_dir = os.path.join(os.path.dirname(__file__), "local_models", model_name.replace("/", "_"))

# 디렉터리가 없으면 생성
os.makedirs(local_dir, exist_ok=True)

print(f"'{model_name}' 모델을 다운로드합니다...")
print(f"저장 위치: {local_dir}")

# 모델 다운로드
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("다운로드 완료!")
