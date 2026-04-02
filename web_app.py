import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional

from src.rag_engine import RAGEngine
from src.ingest_pipeline import ingest_event_stream, run_ingest_pipeline


app = FastAPI(title="Philosophy RAG Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
ALLOWED_UPLOAD_EXT = {".pdf", ".docx", ".json"}
MAX_UPLOAD_BYTES = 120 * 1024 * 1024  # 单文件上限 120MB


class QuestionRequest(BaseModel):
    question: str
    keyword_terms: Optional[List[str]] = None
    source_filters: Optional[List[str]] = None
    auto_extract_keywords: bool = True
    use_hybrid: bool = True
    use_rerank: bool = True
    # 与 src/academic_prompt.py 中 STYLE_* 一致；旧英文 value 仍由 normalize_answer_style 兼容
    answer_style: str = "哲学论述"
    # 供应商：gemini | openai | deepseek
    llm_provider: str = "gemini"
    # 模型 id：如 gemini-2.5-pro / gemini-2.5-flash / o3 / o1 / deepseek-reasoner
    llm_model: str = "gemini-2.5-pro"
    # 是否开启“超长回答”（更高 max_output_tokens，可能更慢且在高峰期更易失败）
    ultra_long_answer: bool = False


class DocItem(BaseModel):
    text: str
    page: Any
    source: str
    chunk_id: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    docs: List[DocItem]
    keyword_hit_docs: List[DocItem] = Field(default_factory=list)
    keyword_source_stats: List[Dict[str, Any]] = Field(default_factory=list)
    profile: str = "quality"
    keywords_used: List[str] = Field(default_factory=list)
    source_filters_used: List[str] = Field(default_factory=list)
    user_terms_used: List[str] = Field(default_factory=list)
    auto_terms_used: List[str] = Field(default_factory=list)
    dropped_terms: List[str] = Field(default_factory=list)
    keyword_query: str = ""
    term_source: str = "question"
    hybrid: bool = False
    reranked: bool = False
    # 与 src/academic_prompt.py 中 STYLE_* 一致；旧英文 value 仍由 normalize_answer_style 兼容
    answer_style: str = "哲学论述"
    answer_model: str = ""
    answer_max_output_tokens: int = 0
    debug: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    total_pages: int
    total_chunks: int


class ProfileRequest(BaseModel):
    profile: str


class ProfileResponse(BaseModel):
    profile: str
    available_profiles: List[str]
    needs_reingest: bool = True


def _safe_unique_upload_path(original_name: str) -> Path:
    name = Path(original_name).name
    if not name or name.strip() != name or ".." in name:
        raise HTTPException(status_code=400, detail="非法文件名")
    suf = Path(name).suffix.lower()
    if suf not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持扩展名：{', '.join(sorted(ALLOWED_UPLOAD_EXT))}",
        )
    stem = Path(name).stem
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    candidate = UPLOAD_DIR / name
    if not candidate.exists():
        return candidate
    for i in range(1, 10000):
        alt = UPLOAD_DIR / f"{stem}_{i}{suf}"
        if not alt.exists():
            return alt
    return UPLOAD_DIR / f"{stem}_{uuid.uuid4().hex[:10]}{suf}"


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    将拖入的文件保存到 data/uploads/，随后与 data 下其余文档一并被 ingest 扫描。
    """
    if not files:
        raise HTTPException(status_code=400, detail="未选择文件")
    saved: List[str] = []
    for f in files:
        if not f.filename:
            continue
        dest = _safe_unique_upload_path(f.filename)
        body = await f.read()
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大（>{MAX_UPLOAD_BYTES // (1024 * 1024)}MB）：{f.filename}",
            )
        dest.write_bytes(body)
        saved.append(dest.name)
    return {"saved": saved, "count": len(saved), "dir": "data/uploads"}


@app.post("/api/ingest", response_model=IngestResponse)
def run_ingest() -> Dict[str, int]:
    """
    在前端点击按钮时运行 ingest 流程（无流式进度，兼容旧客户端）。
    """
    profile = rag_engine.get_profile()
    return run_ingest_pipeline(
        profile,
        rag_engine.params["EMBEDDING_MODEL"],
        data_dir="data",
        reload_sparse_cb=rag_engine.reload_sparse,
    )


@app.post("/api/ingest/stream")
def run_ingest_stream():
    """
    NDJSON 流：每行一个 JSON，含 type=progress|done|error。
    """
    profile = rag_engine.get_profile()
    embed_model = rag_engine.params["EMBEDDING_MODEL"]

    def ndjson_gen():
        try:
            for ev in ingest_event_stream(
                profile,
                embed_model,
                data_dir="data",
                reload_sparse_cb=rag_engine.reload_sparse,
            ):
                yield (json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8")
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield (json.dumps(err, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(
        ndjson_gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/profile", response_model=ProfileResponse)
def get_profile() -> Dict[str, Any]:
    return {
        "profile": rag_engine.get_profile(),
        "available_profiles": rag_engine.get_profile_options(),
        "needs_reingest": True,
    }


@app.post("/api/profile", response_model=ProfileResponse)
def set_profile(req: ProfileRequest) -> Dict[str, Any]:
    ok = rag_engine.switch_profile(req.profile)
    if not ok:
        return {
            "profile": rag_engine.get_profile(),
            "available_profiles": rag_engine.get_profile_options(),
            "needs_reingest": True,
        }
    return {
        "profile": rag_engine.get_profile(),
        "available_profiles": rag_engine.get_profile_options(),
        "needs_reingest": True,
    }


@app.post("/api/answer", response_model=AnswerResponse)
def answer_question(req: QuestionRequest) -> Dict[str, Any]:
    """
    等价于 chat.py 中调用 RAGEngine().answer。
    """
    answer, docs, meta = rag_engine.answer(
        req.question,
        keyword_terms=req.keyword_terms,
        source_filters=req.source_filters,
        auto_extract_keywords=req.auto_extract_keywords,
        use_hybrid=req.use_hybrid,
        use_rerank=req.use_rerank,
        answer_style=req.answer_style,
        llm_provider=req.llm_provider,
        llm_model=req.llm_model,
        ultra_long_answer=req.ultra_long_answer,
    )
    return {
        "answer": answer,
        "docs": docs,
        "keyword_hit_docs": meta.get("keyword_hit_docs", []),
        "keyword_source_stats": meta.get("keyword_source_stats", []),
        "profile": meta.get("profile", "quality"),
        "keywords_used": meta.get("keywords_used", []),
        "source_filters_used": meta.get("source_filters_used", []),
        "user_terms_used": meta.get("user_terms_used", []),
        "auto_terms_used": meta.get("auto_terms_used", []),
        "dropped_terms": meta.get("dropped_terms", []),
        "keyword_query": meta.get("keyword_query", ""),
        "term_source": meta.get("term_source", "question"),
        "hybrid": bool(meta.get("hybrid", False)),
        "reranked": bool(meta.get("reranked", False)),
        "answer_style": meta.get("answer_style", req.answer_style),
        "answer_model": meta.get("answer_model", ""),
        "answer_max_output_tokens": int(meta.get("answer_max_output_tokens", 0)),
        "debug": meta.get("debug", {}),
    }


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

