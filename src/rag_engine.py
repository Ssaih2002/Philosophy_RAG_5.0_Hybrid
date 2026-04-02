import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ANSWER_STYLE_RETRIEVAL_OVERRIDES,
    GEMINI_ANSWER_MODEL,
    GEMINI_FALLBACK_MODELS,
    GEMINI_ANSWER_TEMPERATURE,
    ANSWER_MAX_OUTPUT_TOKENS_DEFAULT,
    ANSWER_MAX_OUTPUT_TOKENS_ULTRA,
    CURRENT_PROFILE,
    PROFILE_SETTINGS,
    MAX_FINAL_K,
    MAX_KEYWORD_HITS,
    MIN_CHUNKS_PER_PRIMARY_SOURCE,
    PRIMARY_SOURCE_COUNT,
)
from .embedder import Embedder
from .vector_store import VectorStore
from .query_expander import expand_query
from .keyword_extractor import extract_keywords_from_question
from .sparse_retriever import SparseRetriever, build_sparse_query
from .hybrid_retrieval import reciprocal_rank_fusion
from .term_merger import merge_terms
from .reranker import CrossEncoderReranker
from .citation import build_context
from .academic_prompt import (
    STYLE_CITE_PATCH,
    STYLE_CONCEPT_MAP,
    build_prompt,
    normalize_answer_style,
)
from .llm_router import generate_answer
from .llm_gemini import generate_with_retry_and_fallback, is_retryable_llm_error


def _split_concept_queries(question: str, merged_terms: List[str]) -> List[str]:
    """概念梳理模式：用关键词与子串检索，避免把整句当作单一扩写问题。"""
    out: List[str] = []
    seen: set = set()
    for t in merged_terms:
        t = (t or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    q = (question or "").strip()
    if q:
        tmp = q
        for sep in [",", "，", "、", "\n", "\r", ";", "；", "|"]:
            tmp = tmp.replace(sep, "\n")
        for line in tmp.split("\n"):
            t = line.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    return out


def replace_source_refs(text, docs):
    """
    Replace occurrences like 'Source 7' or '[Source 7]' in the model output
    with concrete citations '(filename, p. page)' using the retrieved docs.
    """

    def repl(match):
        idx_str = match.group(2)
        try:
            idx = int(idx_str) - 1
        except ValueError:
            return match.group(1)
        if 0 <= idx < len(docs):
            src = docs[idx].get("source", "Unknown")
            page = docs[idx].get("page", "Unknown")
            return f"({src}, p. {page})"
        return match.group(1)

    pattern = re.compile(r"(\[?\s*[Ss]ource\s+(\d+)\s*]?)")
    return pattern.sub(repl, text)


def sanitize_citations(text: str, docs: List[Dict[str, Any]]) -> str:
    """
    Keep only verifiable citations that exist in retrieved docs.
    Any unknown citation '(x, p. y)' is replaced to avoid fabricated references.
    """
    def _norm_src(s: str) -> str:
        return (s or "").strip().lower()

    def _norm_page(p: str) -> str:
        # Be tolerant to common formatting differences:
        # "12", "12 ", "p.12", "12-13", "12–13", "12—13", "12/13"
        s = (p or "").strip().lower()
        s = s.replace("pp.", "").replace("p.", "").replace("p ", "")
        s = s.replace("–", "-").replace("—", "-")
        s = re.sub(r"\s+", "", s)
        return s

    # Build per-source allowed pages to allow range matching.
    allowed_by_source: Dict[str, set] = {}
    for d in docs:
        src0 = str(d.get("source", "Unknown"))
        page0 = str(d.get("page", "Unknown"))
        src = _norm_src(src0)
        page = _norm_page(page0)
        if not src:
            continue
        allowed_by_source.setdefault(src, set()).add(page)

    pattern = re.compile(r"\(([^()]+),\s*p\.\s*([^)]+)\)")

    def _to_int(x: str) -> Optional[int]:
        m = re.search(r"\d+", x or "")
        if not m:
            return None
        try:
            return int(m.group(0))
        except Exception:
            return None

    def repl(match):
        src_raw = match.group(1).strip()
        page_raw = match.group(2).strip()

        src = _norm_src(src_raw)
        page = _norm_page(page_raw)
        allowed_pages = allowed_by_source.get(src, set())

        # If source doesn't match but only one source exists in evidence, align to it.
        if not allowed_pages and len(allowed_by_source) == 1:
            only_src = next(iter(allowed_by_source.keys()))
            allowed_pages = allowed_by_source.get(only_src, set())

        # Exact match
        if page in allowed_pages:
            return f"({src_raw}, p. {page_raw})"

        # Range / composite page tolerant match: accept if any token matches.
        # Examples: "12-13" matches "12" or "13"; "12/13" matches "12" etc.
        tokens = re.split(r"[-/]+", page) if page else []
        tokens = [t for t in tokens if t]
        if any(t in allowed_pages for t in tokens):
            return f"({src_raw}, p. {page_raw})"

        # Substring fallback: "12-13" should match stored "12" (or vice versa)
        for ap in allowed_pages:
            if not ap:
                continue
            if ap in page or page in ap:
                return f"({src_raw}, p. {page_raw})"

        # Numeric tolerance: allow +-1 page drift.
        p_int = _to_int(page)
        if p_int is not None:
            for ap in allowed_pages:
                a_int = _to_int(ap)
                if a_int is None:
                    continue
                if abs(a_int - p_int) <= 1:
                    return f"({src_raw}, p. {page_raw})"

        return "(unverified citation removed)"

    return pattern.sub(repl, text)


def _doc_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": row["text"],
        "page": row.get("page"),
        "source": row.get("source", "Unknown"),
        "chunk_id": row.get("chunk_id"),
    }


def _dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for d in docs:
        cid = d.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(d)
    return out


def _enforce_source_coverage(
    base_docs: List[Dict[str, Any]],
    keyword_hit_docs: List[Dict[str, Any]],
    *,
    per_source_keep: int,
    source_count: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Force balanced source coverage from keyword hits.
    Select top N sources by hit count, then keep up to per_source_keep docs
    from each selected source, and prepend them to the final candidate list.
    """
    if per_source_keep <= 0 or source_count <= 0:
        return base_docs, {"enabled": False}

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for d in keyword_hit_docs:
        src = d.get("source") or "Unknown"
        by_source.setdefault(src, []).append(d)
    if len(by_source) < source_count:
        return base_docs, {"enabled": False, "reason": "insufficient_sources"}

    ranked_sources = sorted(by_source.keys(), key=lambda s: len(by_source[s]), reverse=True)
    selected_sources = ranked_sources[:source_count]

    forced: List[Dict[str, Any]] = []
    forced_counts: Dict[str, int] = {}
    for src in selected_sources:
        picked = by_source[src][:per_source_keep]
        forced.extend(picked)
        forced_counts[src] = len(picked)

    merged = _dedupe_docs(forced + base_docs)
    return merged, {
        "enabled": True,
        "selected_sources": selected_sources,
        "per_source_keep": per_source_keep,
        "forced_counts": forced_counts,
        "forced_total": len(forced),
    }


def _detect_required_language(question: str) -> str:
    q = (question or "").strip()
    if re.search(r"[\u4e00-\u9fff]", q):
        return "Simplified Chinese"
    if re.search(r"[A-Za-z]", q):
        return "English"
    return "same as question"


class RAGEngine:
    def __init__(self):
        print("Initializing RAG system...")
        self.profile = CURRENT_PROFILE
        self.params = dict(PROFILE_SETTINGS[self.profile])
        self.embedder = Embedder(self.params["EMBEDDING_MODEL"])
        self.db = VectorStore(self.profile)
        self.sparse = SparseRetriever(self.profile)
        self.reranker = CrossEncoderReranker(self.params["RERANKER_MODEL"])

    def get_profile(self) -> str:
        return self.profile

    def get_profile_options(self) -> List[str]:
        return list(PROFILE_SETTINGS.keys())

    def switch_profile(self, profile: str) -> bool:
        if profile not in PROFILE_SETTINGS:
            return False
        self.profile = profile
        self.params = dict(PROFILE_SETTINGS[profile])
        self.embedder = Embedder(self.params["EMBEDDING_MODEL"])
        self.db = VectorStore(self.profile)
        self.sparse = SparseRetriever(self.profile)
        self.reranker = CrossEncoderReranker(self.params["RERANKER_MODEL"])
        return True

    def _dense_docs(
        self,
        question: str,
        source_filters: Optional[List[str]] = None,
        *,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        search_k = int(k if k is not None else self.params["SEARCH_K"])
        queries = expand_query(question)
        seen = set()
        ordered: List[Dict[str, Any]] = []
        for q in queries:
            emb = self.embedder.encode([q])[0]
            results = self.db.search(
                emb,
                k=search_k,
                source_filters=source_filters,
            )
            for text, meta in zip(
                results["documents"][0],
                results["metadatas"][0],
            ):
                cid = meta.get("chunk_id")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                ordered.append(
                    {
                        "chunk_id": cid,
                        "text": text,
                        "page": meta.get("page"),
                        "source": meta.get("source", "Unknown"),
                    }
                )
        return ordered

    def _dense_docs_from_queries(
        self,
        queries: List[str],
        source_filters: Optional[List[str]] = None,
        *,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """对多个短查询分别做稠密检索并合并去重（用于概念梳理）。"""
        search_k = int(k if k is not None else self.params["SEARCH_K"])
        ordered: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for q in queries:
            q = (q or "").strip()
            if not q:
                continue
            emb = self.embedder.encode([q])[0]
            results = self.db.search(
                emb,
                k=search_k,
                source_filters=source_filters,
            )
            for text, meta in zip(
                results["documents"][0],
                results["metadatas"][0],
            ):
                cid = meta.get("chunk_id")
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                ordered.append(
                    {
                        "chunk_id": cid,
                        "text": text,
                        "page": meta.get("page"),
                        "source": meta.get("source", "Unknown"),
                    }
                )
        return ordered

    def _effective_retrieval_params(self, style_norm: str) -> Dict[str, Any]:
        p = dict(self.params)
        for key, val in ANSWER_STYLE_RETRIEVAL_OVERRIDES.get(style_norm, {}).items():
            p[key] = val
        return p

    def _resolve_keyword_terms(
        self,
        question: str,
        keyword_terms: Optional[List[str]],
        auto_extract_keywords: bool,
        *,
        max_terms: int = 12,
    ) -> Dict[str, Any]:
        user = [t.strip() for t in (keyword_terms or []) if t and str(t).strip()]
        auto: List[str] = []
        if auto_extract_keywords:
            try:
                auto = extract_keywords_from_question(question)
            except Exception:
                auto = []
        merged = merge_terms(user_terms=user, auto_terms=auto, max_terms=max_terms)

        if merged["user_terms"] and merged["auto_terms"]:
            source = "merged"
        elif merged["user_terms"]:
            source = "user"
        elif merged["auto_terms"]:
            source = "auto"
        else:
            source = "question"

        return {
            "term_source": source,
            **merged,
        }

    def reload_sparse(self):
        self.sparse.reload()

    def retrieve(
        self,
        question: str,
        keyword_terms: Optional[List[str]] = None,
        source_filters: Optional[List[str]] = None,
        auto_extract_keywords: bool = True,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        answer_style: str = "哲学论述",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        style_norm = normalize_answer_style(answer_style)
        rp = self._effective_retrieval_params(style_norm)
        normalized_sources = [
            s.strip() for s in (source_filters or []) if s and str(s).strip()
        ]
        term_budget = 22 if style_norm == STYLE_CONCEPT_MAP else 12
        term_info = self._resolve_keyword_terms(
            question,
            keyword_terms,
            auto_extract_keywords,
            max_terms=term_budget,
        )
        merged_terms = term_info["merged_terms"]
        term_source = term_info["term_source"]
        concept_cq: List[str] = []
        sparse_terms_for_fts = merged_terms

        if style_norm == STYLE_CONCEPT_MAP:
            concept_cq = _split_concept_queries(question, merged_terms)
            seen_ft = set()
            fts_list: List[str] = []
            for t in list(merged_terms) + list(concept_cq):
                tt = (t or "").strip()
                if not tt or tt.lower() in seen_ft:
                    continue
                seen_ft.add(tt.lower())
                fts_list.append(tt)
            sparse_terms_for_fts = fts_list[:36]
            fallback_q = " ".join(concept_cq) if concept_cq else (question or "")
            keyword_query = build_sparse_query(sparse_terms_for_fts, fallback_q)
            dense_list = (
                self._dense_docs_from_queries(
                    concept_cq, source_filters=normalized_sources, k=rp["SEARCH_K"]
                )
                if concept_cq
                else []
            )
        else:
            keyword_query = build_sparse_query(merged_terms, question)
            dense_list = self._dense_docs(question, source_filters=normalized_sources, k=rp["SEARCH_K"])
        dense_ids = [d["chunk_id"] for d in dense_list if d.get("chunk_id")]
        dense_by_id = {d["chunk_id"]: d for d in dense_list if d.get("chunk_id")}

        sparse_ids: List[str] = []
        keyword_hit_docs: List[Dict[str, Any]] = []
        if use_hybrid and self.sparse.is_ready():
            keyword_hits_full = self.sparse.search(
                keyword_query,
                k=MAX_KEYWORD_HITS,
                source_filters=normalized_sources,
            )
            sparse_ids = [
                h["chunk_id"]
                for h in keyword_hits_full[: rp["SPARSE_K"]]
                if h.get("chunk_id")
            ]
            keyword_hit_docs = [_doc_from_row(h) for h in keyword_hits_full]
        keyword_source_counter = Counter(
            (d.get("source") or "Unknown") for d in keyword_hit_docs
        )
        keyword_source_stats = [
            {"source": src, "count": int(cnt)}
            for src, cnt in keyword_source_counter.most_common()
        ]

        hybrid_ok = bool(use_hybrid and self.sparse.is_ready() and sparse_ids)
        if hybrid_ok:
            fused = reciprocal_rank_fusion(
                [dense_ids, sparse_ids], rrf_k=rp["RRF_K"]
            )
            fused_ids = [cid for cid, _ in fused[: rp["HYBRID_TOP_N"]]]
            # Keyword hits first, then fused/dense for semantic coverage.
            top_ids = list(dict.fromkeys(sparse_ids + fused_ids + dense_ids))
        else:
            fused = []
            top_ids = dense_ids[: rp["HYBRID_TOP_N"]]

        docs: List[Dict[str, Any]] = []
        for cid in top_ids:
            row = self.sparse.get_doc(cid)
            if row:
                docs.append(_doc_from_row(row))
            elif cid in dense_by_id:
                docs.append(_doc_from_row(dense_by_id[cid]))

        coverage_meta = {"enabled": False}
        if keyword_hit_docs:
            docs, coverage_meta = _enforce_source_coverage(
                docs,
                keyword_hit_docs,
                per_source_keep=MIN_CHUNKS_PER_PRIMARY_SOURCE,
                source_count=PRIMARY_SOURCE_COUNT,
            )

        baseline_final_k = max(1, min(int(rp["FINAL_K"]), int(MAX_FINAL_K)))
        forced_need = 0
        if coverage_meta.get("enabled"):
            forced_need = int(MIN_CHUNKS_PER_PRIMARY_SOURCE) * int(PRIMARY_SOURCE_COUNT)
        effective_final_k = max(baseline_final_k, forced_need)
        effective_final_k = min(effective_final_k, int(MAX_FINAL_K))
        rerank_enabled = bool(use_rerank and len(docs) > 1)
        rerank_question = question
        if style_norm == STYLE_CONCEPT_MAP:
            rerank_question = (
                " ".join(sparse_terms_for_fts)
                if sparse_terms_for_fts
                else (question or "").strip() or keyword_query
            )
        if rerank_enabled:
            try:
                docs = self.reranker.rerank(
                    question=rerank_question,
                    docs=docs[: max(rp["RERANK_CANDIDATES"], effective_final_k)],
                    top_k=effective_final_k,
                )
            except Exception:
                docs = docs[:effective_final_k]
                rerank_enabled = False
        else:
            docs = docs[:effective_final_k]

        eff_keywords = (
            sparse_terms_for_fts if style_norm == STYLE_CONCEPT_MAP else merged_terms
        )
        meta = {
            "profile": self.profile,
            "keywords_used": eff_keywords,
            "source_filters_used": normalized_sources,
            "user_terms_used": term_info["user_terms"],
            "auto_terms_used": term_info["auto_terms"],
            "dropped_terms": term_info["dropped_terms"],
            "keyword_query": keyword_query,
            "term_source": term_source,
            "hybrid": hybrid_ok,
            "reranked": rerank_enabled,
            "keyword_hit_docs": keyword_hit_docs,
            "keyword_source_stats": keyword_source_stats,
            "coverage_enforced": bool(coverage_meta.get("enabled", False)),
            "answer_style_canonical": style_norm,
            "debug": {
                "profile": self.profile,
                "search_k": rp["SEARCH_K"],
                "final_k": effective_final_k,
                "final_k_configured": rp["FINAL_K"],
                "final_k_cap": MAX_FINAL_K,
                "retrieval_params_effective": {
                    "SEARCH_K": rp["SEARCH_K"],
                    "FINAL_K": rp["FINAL_K"],
                    "SPARSE_K": rp["SPARSE_K"],
                    "HYBRID_TOP_N": rp["HYBRID_TOP_N"],
                    "RERANK_CANDIDATES": rp["RERANK_CANDIDATES"],
                },
                "concept_dense_queries": concept_cq
                if style_norm == STYLE_CONCEPT_MAP
                else [],
                "source_filters": normalized_sources,
                "chroma_collection": self.db.collection_name,
                "chroma_path": self.db.db_path,
                "sparse_db_path": self.sparse.path,
                "dense_top_ids": dense_ids[:12],
                "sparse_top_ids": sparse_ids[:12],
                "fused_top_ids": [cid for cid, _ in fused[:12]],
                "retrieved_before_rerank": len(top_ids),
                "final_docs": len(docs),
                "keyword_hits_count": len(keyword_hit_docs),
                "keyword_hits_cap": MAX_KEYWORD_HITS,
                "keyword_source_stats_top10": keyword_source_stats[:10],
                "coverage": coverage_meta,
            },
        }
        return docs, meta

    def answer(
        self,
        question: str,
        keyword_terms: Optional[List[str]] = None,
        source_filters: Optional[List[str]] = None,
        auto_extract_keywords: bool = True,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        answer_style: str = "哲学论述",
        llm_provider: str = "gemini",
        llm_model: str = "gemini-2.5-pro",
        ultra_long_answer: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        style_norm = normalize_answer_style(answer_style)
        eff_auto = auto_extract_keywords
        if style_norm == STYLE_CONCEPT_MAP:
            eff_auto = False

        docs, meta = self.retrieve(
            question,
            keyword_terms=keyword_terms,
            source_filters=source_filters,
            auto_extract_keywords=eff_auto,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            answer_style=answer_style,
        )
        if not docs:
            msg = (
                "【检索结果为空】当前索引中未检索到与输入相关的文献片段，"
                "无法从语料侧生成引用与脚注。建议：确认已完成 Ingest，检查当前检索配置（quality / fast）与 "
                "`data` 下语料、向量库路径是否一致，或调整关键词 / 限定文件名后重试。"
            )
            meta["retrieval_empty"] = True
            meta["answer_style"] = answer_style
            meta["answer_model"] = ""
            meta["answer_max_output_tokens"] = 0
            return msg, [], meta

        context = build_context(docs)
        required_language = _detect_required_language(question)
        prompt = build_prompt(
            question,
            context,
            required_language=required_language,
            answer_style=answer_style,
        )
        gen_temp = float(GEMINI_ANSWER_TEMPERATURE)
        if normalize_answer_style(answer_style) == STYLE_CITE_PATCH:
            gen_temp = min(0.35, gen_temp)

        max_out = (
            int(ANSWER_MAX_OUTPUT_TOKENS_ULTRA)
            if ultra_long_answer
            else int(ANSWER_MAX_OUTPUT_TOKENS_DEFAULT)
        )

        response_text, model_used = generate_answer(
            prompt=prompt,
            provider=llm_provider,
            model=llm_model,
            temperature=gen_temp,
            max_output_tokens=max_out,
        )
        print(f"Answer generation completed with model={model_used}")
        cleaned_text = replace_source_refs(response_text, docs)
        cleaned_text = sanitize_citations(cleaned_text, docs)
        meta["answer_style"] = answer_style
        meta["answer_model"] = model_used
        meta["answer_max_output_tokens"] = max_out
        return cleaned_text, docs, meta
