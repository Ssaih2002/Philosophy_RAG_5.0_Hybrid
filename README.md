# Philosophy RAG（Hybrid + Rerank）

**一键启动**：本机已安装 **Python 3.11** 时，在仓库根目录双击 **`start_app.bat`**（Windows）或 **`start_app_mac.command`**（macOS）即可；脚本会检查并创建/复用 `.venv`、安装依赖、启动后端并打开前端。首次使用请在 `src/config.py` 中填写 API Key（详见下文「3. 安装与启动」）。

---

本项目是一个面向哲学文本的本地 RAG 系统，支持：

- 稠密检索（Chroma + embedding cosine）
- 稀疏检索（SQLite FTS5 全文索引）
- 混合检索（RRF 融合）
- cross-encoder 重排序（精排）
- 用户关键词 + 自动关键词合并策略
- 检索调试面板（前端显示 dense / sparse / 融合 / rerank 关键信息）
- Web 前端一键 ingest 与问答
- 双配置检索开关（`quality` / `fast`，默认 `quality`）

---

## 1. 本次版本的检索链路

### 1.1 总流程

1. 问题扩写（`query_expander.py`）得到多条稠密查询  
2. 稠密召回（Chroma）得到候选片段列表  
3. 关键词策略：
   - 用户手填关键词（可选）
   - 自动抽词（可选）
   - 两者合并去重（`term_merger.py`）
4. 稀疏召回（SQLite FTS5，`sparse_retriever.py`）  
5. 稠密 + 稀疏做 RRF 融合（`hybrid_retrieval.py`）  
6. cross-encoder 对融合候选精排（`reranker.py`）  
7. 取最终 `FINAL_K` 片段构造上下文，调用所选 LLM（Gemini / OpenAI / DeepSeek）生成回答  
8. 按 `(source, p. page)` 样式输出引用
9. 输出前对引用做“可核验清洗”：不在检索证据中的 `(source, p. page)` 会被移除

### 1.2 三个关键能力

- **cross-encoder 重排序**：提升最终 Top-K 引用质量  
- **SQLite FTS5 稀疏检索**：增强术语/专名的字面召回  
- **用户词 + 自动词合并**：兼顾可控性与召回率

---

## 2. 目录结构（核心）

```text
philosophy_rag5.0_Hybrid/   # 示例根目录名，可按你的克隆路径调整
├─ start_app.bat            # Windows 一键：创建 .venv、装依赖、起后端、打开前端
├─ start_app_mac.command    # macOS 一键启动
├─ run_backend.bat          # 仅启动 uvicorn（供 start_app 调用）
├─ ingest.py
├─ ingest_single_tmp.py     # 单目录临时 ingest（如 tmp profile）
├─ merge_profile.py         # 合并两个 profile 的向量/稀疏索引
├─ merge_tmp_to_quality.bat
├─ web_app.py
├─ frontend.html
├─ chat.py
├─ compare.py
├─ LDA.py
├─ requirements.txt
├─ tools/
│  └─ ensure_torch_accel.py # （可选）检测到 NVIDIA 时尝试安装 CUDA 版 PyTorch
├─ data/
│  ├─ chroma_db_quality/ # quality profile 稠密向量库（ingest 后生成）
│  ├─ chroma_db_fast/    # fast profile（同上）
│  ├─ sparse_fts_quality.db
│  ├─ sparse_fts_fast.db
│  ├─ pdf/               # 语料 PDF 等（按需）
│  └─ uploads/           # Web 上传文件目录
└─ src/
   ├─ config.py
   ├─ rag_engine.py
   ├─ vector_store.py
   ├─ sparse_retriever.py
   ├─ hybrid_retrieval.py
   ├─ reranker.py
   ├─ term_merger.py
   ├─ keyword_extractor.py
   ├─ query_expander.py
   ├─ citation.py
   ├─ academic_prompt.py
   ├─ document_loader.py
   ├─ semantic_chunker.py
   └─ embedder.py
```

---

## 3. 安装与启动

### 3.1 安装依赖

```powershell
# 建议固定使用 Python 3.11 创建项目虚拟环境
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果你之前用 3.14 等建过旧环境，建议先删除旧 `venv` / `.venv` 再重建，避免包冲突：

```powershell
deactivate
Remove-Item -Recurse -Force .\venv
Remove-Item -Recurse -Force .\.venv
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python --version
```

`python --version` 显示 `3.11.x` 即成功。

### 3.2 配置 API Key 与代理

编辑 `src/config.py`：

- **API Key**：`GEMINI_API_KEY`、`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`（按需填写）
- **代理（可选）**：`HTTP_PROXY_URL`、`HTTPS_PROXY_URL`、`SOCKS_PROXY_URL`（例如本机 v2ray 常见 `http://127.0.0.1:33210`；SOCKS 需已安装 `httpx[socks]`，见 `requirements.txt`）

**安全提示**：若仓库会推送到 GitHub 等公开位置，请**勿**在代码中保留真实 Key；应改用环境变量或私有配置文件，并在泄露后**立即轮换** Key。

### 3.3 检索配置开关（默认 bge-m3 多语言）

在 `src/config.py` 中：

```python
RETRIEVAL_PROFILE = "quality"  # "quality" | "fast"
```

- `quality`（默认）：
  - `EMBEDDING_MODEL = "BAAI/bge-m3"`
  - `RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"`
  - 面向中/英/德混合语料，质量优先
- `fast`：
  - `EMBEDDING_MODEL = "BAAI/bge-small-en"`
  - `RERANKER_MODEL = "BAAI/bge-reranker-base"`
  - 速度优先，跨语言鲁棒性较弱

你现在也可以在前端直接切换（步骤 1 区域的“检索配置”下拉框，选择即切换）。  
切换后后端会立即应用新 profile，但为了保证索引与 embedding 一致，**必须重新运行一次 Ingest**。

### 3.3.1 不同 profile 的向量库是否互通？

默认不互通（也不建议互通），因为不同 embedding 模型输出的向量空间不同。  
本项目现已改成“按 profile 分库命名”：

- Chroma：
  - `quality` -> `data/chroma_db_quality`，集合名 `philosophy_quality`
  - `fast` -> `data/chroma_db_fast`，集合名 `philosophy_fast`
- FTS5：
  - `quality` -> `data/sparse_fts_quality.db`
  - `fast` -> `data/sparse_fts_fast.db`

这样可以在同一项目中并行保留两套索引，互不覆盖，便于对比。

### 3.4 运行后端

```powershell
python -m uvicorn web_app:app --reload --host 127.0.0.1 --port 8000
```

### 3.5 打开前端

直接双击 `frontend.html`（或浏览器打开），然后：

1. 先点击“运行 Ingest”
2. 再输入问题点击“提问”

### 3.6 一键启动（Windows / macOS）

- Windows：双击 `start_app.bat`
  - 会自动创建 `.venv`、安装依赖、启动后端并打开 `frontend.html`
  - 若检测到 NVIDIA 环境，会尝试自动安装 PyTorch CUDA wheel（失败则继续使用 CPU 版）
- macOS：双击 `start_app_mac.command`
  - 首次运行可能需要在“系统设置→隐私与安全性”允许该脚本执行
  - macOS 不支持 CUDA；Apple Silicon 上 PyTorch 会优先使用 MPS（若可用）

---

## 4. ingest 说明（重要）

`ingest.py` 与 `/api/ingest` 均执行以下流程：

1. 加载 `data/` 下文档（pdf/docx/json）
2. 语义分块
3. embedding 编码
4. 重建 Chroma 集合并写入稠密向量
5. 重建 SQLite FTS5 稀疏索引（按 profile 命名，如 `data/sparse_fts_quality.db`）

注意：本版本 `chunk_id` 采用稳定编号 `chunk_0 ... chunk_n-1`，稠密与稀疏索引通过 `chunk_id` 对齐。

---

## 5. 问答接口（`POST /api/answer`）

### 5.1 请求体

```json
{
  "question": "康德如何区分现象界与物自身？",
  "keyword_terms": ["noumenon", "thing in itself"],
  "source_filters": ["Kant_CPR.pdf", "Hegel_Phenomenology.pdf"],
  "auto_extract_keywords": true,
  "use_hybrid": true,
  "use_rerank": true,
  "answer_style": "哲学论述",
  "ultra_long_answer": false,
  "llm_provider": "gemini",
  "llm_model": "gemini-2.5-pro"
}
```

### 5.2 字段说明

- `answer_style`：回答风格，与 `src/academic_prompt.py` 中风格一致（如 `哲学论述`、`文献综述` 等）
- `ultra_long_answer`：是否开启超长输出上限（`false` 时约 12288 tokens，`true` 时约 24576；更长更易慢、更易触发限流/断连，见 `src/config.py` 中 `ANSWER_MAX_OUTPUT_TOKENS_*`）
- `keyword_terms`：用户手填术语（可选）
- `source_filters`：按文件名限定检索范围（可选，多个文件名）
- `auto_extract_keywords`：是否自动抽取术语
- `use_hybrid`：是否启用稠密+稀疏混合检索
- `use_rerank`：是否启用 cross-encoder 精排
- `llm_provider`：回答模型供应商，支持 `gemini` / `openai` / `deepseek`
- `llm_model`：具体模型 ID，例如：
  - Gemini：`gemini-2.5-pro` / `gemini-2.5-flash`
  - OpenAI：`gpt-5.1` / `gpt-5-mini`（当选择 `gpt-5.1` 且失败 3 次，会自动降级到 `gpt-5-mini`）
  - DeepSeek：`deepseek-reasoner`

### 5.3 响应新增元信息

- `answer_model`：本次实际用于生成回答的模型标识（如 `gemini:gemini-2.5-flash`）
- `answer_max_output_tokens`：本次请求使用的输出 token 上限
- `keywords_used`：最终实际用于稀疏查询的关键词
- `source_filters_used`：本次实际生效的文件名过滤列表
- `user_terms_used`：用户术语
- `auto_terms_used`：自动抽取术语
- `dropped_terms`：去重/截断后被丢弃术语
- `keyword_query`：发送给 FTS5 的查询表达式
- `term_source`：`question` / `user` / `auto` / `merged`
- `hybrid`：本次是否实际走了混合检索
- `reranked`：本次是否完成重排序
- `debug`：调试字段（dense/sparse/fused id 列表、最终条数等）

### 5.4 配置切换接口

- `GET /api/profile`：获取当前 profile 与可选 profile
- `POST /api/profile`：切换 profile

请求示例：

```json
{ "profile": "quality" }
```

### 5.5 前端限定文件名窗口

步骤 2 中新增“限定文件名（可选）”输入框：  
当你输入一个或多个文件名后，系统仅在这些文件对应的片段中进行稠密+稀疏检索并作答。  
适用于语料库很大时的定向问答。

---

## 6. 检索参数（`src/config.py`）

默认值以仓库内 **`PROFILE_SETTINGS`（`quality` / `fast`）** 与 **`ANSWER_STYLE_RETRIEVAL_OVERRIDES`（按回答风格覆盖）** 为准，请勿以本文旧版示例数字为准。

调参方向简述：

- `SEARCH_K` 增大：稠密召回更广，但更慢
- `SPARSE_K` 增大：术语召回更广，但噪声可能上升
- `RERANK_CANDIDATES` 增大：精排更准，但速度下降
- `FINAL_K`：最终喂给 LLM 的证据片段数量

补充：前端调试面板会显示当前 profile，便于确认是否已切换成功。

---

## 7. requirements 说明

本项目依赖如下（见 `requirements.txt`）：

- `chromadb`：稠密向量库
- `sentence-transformers`：embedding 与 cross-encoder 模型加载
- `langchain-text-splitters`：语义分块
- `google-genai`：Gemini SDK
- `httpx[socks]`：OpenAI / DeepSeek HTTP 调用；SOCKS 代理需此 extras
- `pymupdf`、`python-docx`：文档读取
- `nltk`、`jieba`、`gensim`、`pyLDAvis`：NLP 与 LDA
- `matplotlib`、`seaborn`、`pandas`、`networkx`：可视化与统计
- `torch`、`transformers`：深度学习底层
- `fastapi`、`uvicorn[standard]`：后端服务（含 `watchfiles` 等，`--reload` 更稳）
- `python-multipart`：`/api/upload` 等多部分表单解析（FastAPI 官方建议显式安装）

说明：稀疏检索已迁移为 SQLite FTS5，不再依赖 `rank-bm25`。

---

## 8. 常见问题

### Q1：问答时报重排序模型加载慢

首次启用 `use_rerank=true` 时会下载并加载 `RERANKER_MODEL`，属于正常现象。可先关闭重排序验证主链路，再开启。

### Q2：混合检索没有生效

请先执行 ingest，确认对应 profile 的 SQLite 已生成，例如 `data/sparse_fts_quality.db`（或 fast：`data/sparse_fts_fast.db`）；若未生成，检查写权限与日志报错。

### Q3：为什么重建 ingest 会清空旧向量库

当前实现是“全量重建”策略，避免 `chunk_id` 不一致导致稠密/稀疏错配。

### Q4：如何验证“引用是客观的”

当前实现包含两层保护：

1. Prompt 强约束：模型只能引用上下文里的 `Cite as` 格式。  
2. 后处理清洗：回答输出后会校验 `(source, p. page)` 是否出现在本次检索证据中；不在证据集合中的引用会被替换为 `(unverified citation removed)`，从而避免伪造引用。

---

## 9. 单文献补索引与合并（可选）

当你怀疑某一篇文献未被充分索引时，可以对该文献单独 ingest 到临时 profile，然后合并进主库。

### 9.1 准备单文献目录

将要补索引的 PDF 放到项目根目录的 `data_single/` 下（可包含子目录）。

### 9.2 运行临时 ingest（tmp）

```powershell
.\.venv\Scripts\python.exe ingest_single_tmp.py --profile tmp --data-dir data_single --embedding-model BAAI/bge-m3
```

### 9.3 合并到 quality 主库

```powershell
.\.venv\Scripts\python.exe merge_profile.py tmp quality
```

或直接双击：

- `merge_tmp_to_quality.bat`

### Q5：前端哪里看调试与引用原文

- 最右侧可隐藏“检索调试面板”：可查看关键词合并结果、dense/sparse/fused IDs、是否 rerank。  
- 最右侧“引用文献片段”窗口：查看本次引用片段列表并点击查看全文。

