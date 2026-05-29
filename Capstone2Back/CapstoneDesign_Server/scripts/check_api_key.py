import os
import sys
import io
from openai import OpenAI
from dotenv import load_dotenv

# Prevent UnicodeEncodeError on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_key():
    print("--- OpenAI API 키 유효성 검사기 ---")
    
    # 1. .env 파일에서 API 키 로드
    print("1. .env 파일에서 키를 로드합니다...")
    load_dotenv()
    
    # 2. 키 변수 가져오기
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 3. 키 존재 여부 1차 확인
    if not api_key:
        print("\n" + "!"*30)
        print("❌ 오류: .env 파일에서 'OPENAI_API_KEY' 변수를 찾을 수 없습니다.")
        print("   파일 이름이 '.env'가 맞는지, 변수 이름에 오타가 없는지 확인하세요.")
        print("!"*30)
        return

    if not api_key.startswith("sk-"):
        print("\n" + "!"*30)
        print(f"❌ 오류: 키가 'sk-'로 시작하지 않습니다. (현재 값: {api_key[:10]}...)")
        print("   OpenAI API 키는 항상 'sk-'로 시작합니다. 키를 다시 복사하세요.")
        print("!"*30)
        return

    print(f"   > 키 발견: {api_key[:5]}...{api_key[-4:]}")

    # 4. OpenAI API에 연결 시도
    try:
        print("2. OpenAI 서버에 연결을 시도합니다...")
        client = OpenAI(api_key=api_key)
        
        # 5. 인증 테스트를 위한 가장 간단한 API 호출 (모델 목록 가져오기)
        print("3. 키 인증을 위해 모델 목록을 요청합니다...")
        client.models.list()
        
        # 6. 성공
        print("\n" + "="*30)
        print("✅ 성공! API 키가 유효하며 정상적으로 작동합니다.")
        print("   이제 main.py를 실행해도 됩니다.")
        print("="*30)

    except Exception as e:
        # 7. 실패
        print("\n" + "!"*30)
        print("❌ 실패! API 키가 유효하지 않거나 다른 오류가 발생했습니다.")
        print("\n오류 상세 정보:")
        print(e)
        print("\n해결책: OpenAI 사이트에서 API 키를 새로 발급받아 .env 파일에 붙여넣으세요.")
        print("!"*30)

if __name__ == "__main__":
    check_key()
