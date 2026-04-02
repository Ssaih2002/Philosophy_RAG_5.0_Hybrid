from typing import Any, List, Optional, Sequence, Union

from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        print("Loading embedding model...")
        self.model = SentenceTransformer(self.model_name)

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        show_progress_bar: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)
        if show_progress_bar is None:
            show_progress_bar = len(texts) > 1
        emb = self.model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )
        return emb.tolist()
