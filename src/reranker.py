from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            print("Loading reranker model...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, question: str, docs: List[Dict], top_k: int) -> List[Dict]:
        if not docs:
            return docs
        model = self._get_model()
        pairs = [[question, d.get("text", "")] for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(docs, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        out = []
        for doc, score in ranked[:top_k]:
            item = dict(doc)
            item["rerank_score"] = float(score)
            out.append(item)
        return out
