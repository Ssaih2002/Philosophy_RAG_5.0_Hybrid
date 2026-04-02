GEMINI_API_KEY = ""
OPENAI_API_KEY = ""    # 例如：sk-...
DEEPSEEK_API_KEY = ""  # 例如：sk-...
GEMINI_ANSWER_MODEL = "gemini-2.5-pro"
# 辅助模型用于问题扩写等检索侧任务：优先稳定/速度
GEMINI_AUX_MODEL = "gemini-2.5-flash"
# 主模型不可用（503/429）时按顺序降级尝试，建议从强到快
GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash",
]
# query expander（问题扩写）失败时的兜底模型
GEMINI_AUX_FALLBACK_MODEL = "gemini-2.5-flash"
# 单模型最大重试次数（含首次请求，>=1）
GEMINI_RETRY_MAX_ATTEMPTS = 3
# 指数退避基数秒（实际等待 = base * 2^(attempt-1) + 抖动）
GEMINI_RETRY_BASE_SECONDS = 1.2
# 每次重试的最大随机抖动秒，避免并发雪崩
GEMINI_RETRY_JITTER_SECONDS = 0.6
# 回答输出上限：默认更稳（12288）；手动开启“超长回答”才使用 24576
ANSWER_MAX_OUTPUT_TOKENS_DEFAULT = 12288
ANSWER_MAX_OUTPUT_TOKENS_ULTRA = 24576
GEMINI_ANSWER_TEMPERATURE = 0.7

# --- 其他供应商（key 直接在此处配置）---

# 可选：代理（v2ray / Clash 等）
# - HTTP(S) 代理（常见 33210）
# - SOCKS5 代理（常见 33211；会写入 ALL_PROXY，很多库会使用）
# 留空则直连。
HTTP_PROXY_URL = "http://127.0.0.1:33210"
HTTPS_PROXY_URL = "http://127.0.0.1:33210"
SOCKS_PROXY_URL = ""

# OpenAI Responses API base
OPENAI_BASE_URL = "https://api.openai.com/v1"

# DeepSeek OpenAI-compatible base
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# OpenAI 主力模型（按你当前可用模型）
OPENAI_MODEL_PRIMARY = "gpt-5.1"
OPENAI_MODEL_SECONDARY = "gpt-5-mini"

# OpenAI 侧：并发与重试（避免 429 雪崩）
# - 并发过高时会更容易触发 429；Web 端多用户同时问答建议设为 1~3
OPENAI_MAX_CONCURRENCY = 2
# - OpenAI 429/503 等可恢复错误的最大重试次数（含首次请求，>=1）
OPENAI_RETRY_MAX_ATTEMPTS = 5
# - 指数退避基数秒（实际等待 = base * 2^(attempt-1) + 抖动；同时会尊重 Retry-After）
OPENAI_RETRY_BASE_SECONDS = 1.5
# - 每次重试的最大随机抖动秒
OPENAI_RETRY_JITTER_SECONDS = 0.8
# - 单次 sleep 上限（秒），避免等待过久卡死请求
OPENAI_RETRY_MAX_SLEEP_SECONDS = 30.0

# DeepSeek 主力模型
DEEPSEEK_MODEL_PRIMARY = "deepseek-reasoner"

CHROMA_PATH = "data/chroma_db"
SPARSE_DB_PATH = "data/sparse_fts.db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Avoid unbounded context expansion; allow large but controlled values.
MAX_FINAL_K = 200
# Full keyword-hit list cap for side panel and API payload safety.
MAX_KEYWORD_HITS = 2000
# When multiple sources are involved, keep at least N chunks per source.
MIN_CHUNKS_PER_PRIMARY_SOURCE = 15
# Number of primary sources to enforce in balanced coverage mode.
PRIMARY_SOURCE_COUNT = 2

RETRIEVAL_PROFILE = "quality"  # "quality" | "fast"

# 在「检索 profile」基础上，按回答风格再覆盖（键与 academic_prompt 中文 value 一致）
ANSWER_STYLE_RETRIEVAL_OVERRIDES = {
    "概念梳理": {
        "FINAL_K": 28,
        "SEARCH_K": 36,
        "SPARSE_K": 64,
        "HYBRID_TOP_N": 56,
        "RERANK_CANDIDATES": 52,
    },
    "文献综述": {
        "FINAL_K": 18,
        "SEARCH_K": 32,
        "SPARSE_K": 52,
        "HYBRID_TOP_N": 44,
        "RERANK_CANDIDATES": 40,
    },
    "哲学论述": {
        "FINAL_K": 16,
        "SEARCH_K": 30,
        "SPARSE_K": 48,
        "HYBRID_TOP_N": 40,
        "RERANK_CANDIDATES": 36,
    },
    "盲审审稿": {
        "FINAL_K": 14,
        "SEARCH_K": 28,
        "SPARSE_K": 44,
        "HYBRID_TOP_N": 38,
        "RERANK_CANDIDATES": 34,
    },
    "引文补注": {
        "FINAL_K": 20,
        "SEARCH_K": 32,
        "SPARSE_K": 52,
        "HYBRID_TOP_N": 44,
        "RERANK_CANDIDATES": 40,
    },
    "学术分析": {
        "FINAL_K": 14,
        "SEARCH_K": 26,
        "SPARSE_K": 40,
        "HYBRID_TOP_N": 34,
        "RERANK_CANDIDATES": 30,
    },
    "简洁作答": {
        "FINAL_K": 8,
        "SEARCH_K": 18,
        "SPARSE_K": 28,
        "HYBRID_TOP_N": 24,
        "RERANK_CANDIDATES": 22,
    },
}

PROFILE_SETTINGS = {
    "quality": {
        # default profile: multilingual quality (CN/EN/DE)
        "EMBEDDING_MODEL": "BAAI/bge-m3",
        "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
        "SEARCH_K": 22,
        "FINAL_K": 10,
        "SPARSE_K": 34,
        "HYBRID_TOP_N": 28,
        "RRF_K": 60,
        "RERANK_CANDIDATES": 26,
    },
    "fast": {
        # lighter profile: faster but less multilingual robustness
        "EMBEDDING_MODEL": "BAAI/bge-small-en",
        "RERANKER_MODEL": "BAAI/bge-reranker-base",
        "SEARCH_K": 12,
        "FINAL_K": 5,
        "SPARSE_K": 18,
        "HYBRID_TOP_N": 14,
        "RRF_K": 60,
        "RERANK_CANDIDATES": 14,
    },
}

CURRENT_PROFILE = RETRIEVAL_PROFILE if RETRIEVAL_PROFILE in PROFILE_SETTINGS else "quality"
_P = PROFILE_SETTINGS[CURRENT_PROFILE]

EMBEDDING_MODEL = _P["EMBEDDING_MODEL"]
RERANKER_MODEL = _P["RERANKER_MODEL"]
SEARCH_K = _P["SEARCH_K"]
FINAL_K = _P["FINAL_K"]
SPARSE_K = _P["SPARSE_K"]
HYBRID_TOP_N = _P["HYBRID_TOP_N"]
RRF_K = _P["RRF_K"]
RERANK_CANDIDATES = _P["RERANK_CANDIDATES"]