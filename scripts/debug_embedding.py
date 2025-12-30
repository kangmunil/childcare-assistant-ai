import os
import requests
import json
from dotenv import load_dotenv

# .env 로드
load_dotenv(override=True)

def test_openrouter_embedding():
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = "google/gemini-embedding-001"
    url = "https://openrouter.ai/api/v1/embeddings"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "input": "안녕하세요, 테스트 문장입니다."
    }
    
    print(f"Calling OpenRouter with model: {model}")
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    print(f"Status Code: {response.status_code}")
    try:
        res_json = response.json()
        print("Response Structure Keys:", res_json.keys())
        if 'data' in res_json:
            print("Dimension:", len(res_json['data'][0]['embedding']))
        else:
            print("Full Response:", res_json)
    except Exception as e:
        print("Error parsing response:", e)

if __name__ == "__main__":
    test_openrouter_embedding()
