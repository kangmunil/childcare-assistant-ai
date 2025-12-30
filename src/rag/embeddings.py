import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from loguru import logger

class OpenRouterEmbeddings(Embeddings):
    """
    Custom Embeddings class for OpenRouter.
    Ensures compatibility with non-OpenAI models like Gemini.
    """
    def __init__(self, model: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def _embed(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # OpenRouter suggests adding these headers
        headers["HTTP-Referer"] = "https://github.com/kangmunil/childcare-assistant"
        headers["X-Title"] = "Childcare Assistant AI"

        url = f"{self.base_url}/embeddings"
        data = {
            "model": self.model,
            "input": texts
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            res_json = response.json()
            
            # Extract embeddings
            # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
            embeddings = [item["embedding"] for item in res_json["data"]]
            return embeddings
        except Exception as e:
            logger.error(f"OpenRouter Embedding Error: {e}")
            if response is not None:
                logger.error(f"Response: {response.text}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # OpenRouter handles batches, but we might want to split if too large
        # For now, simple pass-through
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
