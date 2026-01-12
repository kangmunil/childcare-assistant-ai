from src.rag.childcare_agent import ChildcareAgent
from dotenv import load_dotenv
import os
import sys

def main():
    # 환경 변수 로드
    env_path = os.path.join(os.getcwd(), '.env')
    load_dotenv(dotenv_path=env_path, override=True)
    
    print("\n" + "="*50)
    print("👶 육아 헬퍼 '베이비봇'과 대화를 시작합니다.")
    print("종료하시려면 '종료' 또는 'exit'를 입력하세요.")
    print("="*50 + "\n")

    try:
        agent = ChildcareAgent()
    except Exception as e:
        print(f"에이전트 초기화 실패: {e}")
        return

    while True:
        user_input = input("\n[부모님]: ")
        
        if user_input.lower() in ['종료', 'exit', 'quit']:
            print("\n베이비봇을 종료합니다. 육아 화이팅하세요! 😊")
            break
            
        if not user_input.strip():
            continue

        try:
            print("\n[베이비봇]: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
        except Exception as e:
            print(f"\n에러가 발생했습니다: {e}")

if __name__ == "__main__":
    main()

