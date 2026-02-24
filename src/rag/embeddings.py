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
        
        # DEBUG LOGGING
        logger.info(f"OpenRouter Request - Model: {self.model}, URL: {url}")
        logger.info(f"Input batch size: {len(texts)}")
        if texts:
            logger.info(f"First input sample: {texts[0][:100]}...")

        
        data = {
            "model": self.model,
            "input": texts
        }

        response = None
        
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
        # OpenRouter handles batches, but we manually batch to be safe and debug
        batch_size = 10
        all_embeddings = []
        
        logger.info(f"Starting batch embedding for {len(texts)} texts with batch_size={batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                embeddings = self._embed(batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Batch {i//batch_size} failed: {e}")
                raise e
                
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
