from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_id_lists: List[List[str]],
    rrf_k: int = 60,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
