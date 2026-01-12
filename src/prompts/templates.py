import yaml
import os
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class PromptLoader:
    """
    YAML 파일에서 프롬프트를 로드하여 관리하는 클래스
    """
    
    def __init__(self, prompt_path: str = "src/prompts/system.yaml"):
        # 절대 경로로 변환
        self.prompt_path = os.path.join(os.getcwd(), prompt_path)
        self.prompts = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        try:
            with open(self.prompt_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return {}

    def get_system_prompt(self, key: str = "default") -> str:
        """
        YAML 구조를 평문 텍스트로 변환하여 시스템 프롬프트를 생성합니다.
        """
        data = self.prompts.get(key, {})
        if not data:
            return "당신은 도움이 되는 AI 어시스턴트입니다."

        role = data.get("role", "")
        
        guidelines = "\n".join([f"- {item}" for item in data.get("guidelines", [])])
        
        tools_list = []
        for item in data.get("tools_description", []):
            if isinstance(item, dict):
                for k, v in item.items():
                    tools_list.append(f"- {k}: {v}")
            else:
                tools_list.append(f"- {item}")
        tools = "\n".join(tools_list)

        full_prompt = f"""{role}

[지침]
{guidelines}

[사용 가능한 도구]
{tools}
"""
        return full_prompt.strip()

# 싱글톤 인스턴스
prompt_loader = PromptLoader()
