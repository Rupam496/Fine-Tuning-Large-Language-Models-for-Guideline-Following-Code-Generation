import os

import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings


load_dotenv()


class JinaCodeEmbeddings(Embeddings):
    def __init__(
        self,
        model="jina-embeddings-v2-base-code",
    ):
        self.model = model
        self.api_key = os.getenv("JINA_API_KEY")
        self.url = "https://api.jina.ai/v1/embeddings"

        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY is not set in the environment."
            )

    def _embed(self, texts):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(
            self.url,
            headers=headers,
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

        return [
            item["embedding"]
            for item in result["data"]
        ]

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]