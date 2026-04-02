"""
回答风格（answer_style）约定：

- API / 前端请优先使用下列「中文 value」（名称日后若要改，只改此处常量即可）。
- 仍接受旧版英文 value，由 normalize_answer_style() 统一映射。

已在前端下架、代码中仍兼容的风格：学术分析、简洁作答（见 DEPRECATED_STYLES）。
"""

# --- 主模式（推荐给 API / 前端的中文 value）---
STYLE_PHILOSOPHER = "哲学论述"
STYLE_REVIEW = "盲审审稿"
STYLE_CITE_PATCH = "引文补注"
STYLE_CONCEPT_MAP = "概念梳理"
STYLE_LITERATURE_REVIEW = "文献综述"

# --- 已弃用（前端不再展示；保留兼容与 prompt 分支）---
STYLE_ACADEMIC = "学术分析"  # 对应旧 english: academic
STYLE_CONCISE = "简洁作答"  # 对应旧 english: concise

DEPRECATED_STYLES = frozenset({STYLE_ACADEMIC, STYLE_CONCISE})


def normalize_answer_style(answer_style: str) -> str:
    """将英文或别名统一为内部使用的中文 canonical style（先匹配专名，避免误映射）。"""
    if not answer_style:
        return STYLE_PHILOSOPHER
    s = str(answer_style).strip()
    key = s.lower()

    if s in (STYLE_CONCEPT_MAP, "关键词谱系") or key == "concept_map":
        return STYLE_CONCEPT_MAP
    if (
        s in (STYLE_LITERATURE_REVIEW, "综述", "文献回顾")
        or key in ("literature_review", "lit_review")
    ):
        return STYLE_LITERATURE_REVIEW
    if (
        s in (STYLE_CITE_PATCH, "仅补脚注", "补引用")
        or key == "cite_patch"
    ):
        return STYLE_CITE_PATCH
    if s in (STYLE_REVIEW, "盲审") or key == "review":
        return STYLE_REVIEW
    if key == "academic" or s == STYLE_ACADEMIC:
        return STYLE_ACADEMIC
    if key == "concise" or s == STYLE_CONCISE:
        return STYLE_CONCISE
    if (
        s in (STYLE_PHILOSOPHER, "哲学沉思者")
        or key == "philosophical"
    ):
        return STYLE_PHILOSOPHER

    return STYLE_PHILOSOPHER


def _length_and_coverage_block(style: str) -> str:
    """强制长答与多脚注（中文汉字规模指正文，不含脚注列表）。"""
    if style == STYLE_CONCEPT_MAP:
        return """
## 篇幅与覆盖面（概念梳理 — 强制，优先级高于「简洁」类暗示）

- **正文汉字**：不少于 **4000 字**，目标区间 **4500–6500 字**（`## 脚注` 另计）。明显短于此视为未完成，须继续扩写至达标。
- **脚注**：至少 **22** 条有效条目；须优先使用 Sources 中不同出处、不同页码，**尽量覆盖多数 excerpt**；同一页可多条若引用点不同。
- **结构**：至少 **5** 个二级标题（`## …`），每节内多段、每段尽量含 **1–2 处** `（引文或严密转写）[n]`。
- **单文本 / 多文本**：按计划分节写全；多文本时必须专设一节写 **用法差异与概念谱系**。
- 禁止仅列提纲、禁止「限于篇幅」式省略；Sources 中有利证据须尽量纳入。
"""
    if style == STYLE_LITERATURE_REVIEW:
        return """
## 篇幅与覆盖面（文献综述 — 强制）

- **正文汉字**：不少于 **3600 字**，目标区间 **4200–5600 字**（`## 脚注` 另计）。
- **脚注**：至少 **16** 条，尽量覆盖不同来源与不同页码，避免只围绕单一出处重复注释。
- **结构**：至少 **5** 个二级标题（`## …`），至少包含：研究问题与范围、研究脉络/分期、核心争论、方法与证据评估、研究缺口与未来议题。
- **方法要求**：不只罗列观点，必须比较立场差异、论证强弱与证据类型。
"""
    if style == STYLE_PHILOSOPHER:
        return """
## 篇幅与覆盖面（哲学论述 — 强制）

- **正文汉字**：不少于 **3200 字**，目标 **3800–5200 字**（脚注另计）。
- **脚注**：至少 **14** 条，覆盖多段论证；核心论断须有引文支撑。
- 多段落展开论证，不得以短答代替。
"""
    if style == STYLE_REVIEW:
        return """
## 篇幅（盲审审稿）

- 总篇幅宜 **3000 汉字以上**（若用户来稿极短则从宽，但仍须写全审稿维度）。
- 引用用户稿或语料处仍用 `[n]` 脚注体例；脚注**不少于 8 条**（若语料极贫乏则可从宽并说明）。
"""
    if style == STYLE_CITE_PATCH:
        return """
## 篇幅（引文补注）

- 正文长度**以用户来稿为准**，不因追求长答而增删用户字句。
- 脚注条数随可核引用数量自然增长；**凡可溯源处尽量加注**。
"""
    if style in (STYLE_ACADEMIC, STYLE_CONCISE):
        return """
## 篇幅

- 学术分析 / 简洁模式：正文目标 **2200–3500 字**；脚注**不少于 10 条**（简洁模式亦不得以过少脚注敷衍）。
"""
    return ""


def _footnote_rules_block() -> str:
    return """
## 引用与脚注格式（所有模式强制一致）

你必须采用中文学术论文常见的「随文上标式脚注」习惯，在 Markdown 中统一为：

0. **编号一致性（强制）**：脚注必须从 `[1]` 开始，按正文出现顺序严格递增；严禁跳号、重复或回溯。输出前自检：正文最后一个编号是否等于脚注列表条数。

1. **正文**：在运用或引用文献观点之处，先给出原文摘录或高度依字面之转写，再紧跟脚注编号，格式严格为：
   `（……引文或转写……）[n]`
   其中 `[n]` 使用**半角方括号与阿拉伯数字**（例如 `[1]`、`[2]`），紧接引文闭括号之后，中间可不加空格。

2. **文末脚注列表**：正文全部结束后，另起一行使用二级标题：`## 脚注`，然后逐条列出：
   `[n]（文献文件名或 Cite as 所示名称, p. 页码）` 后接该条目的说明（出处、所引片段与正文中何句对应等）。
   脚注条中的文献名、页码**必须**来自下方 Sources 中提供的 `Cite as: (…, p. …)`，不得虚构。

3. **禁止**：在正文中单独使用 `(filename, p. x)` 而不通过 `[n]` 指向脚注表（除非该模式另有说明）；禁止使用 `Source 1`、`[Source 3]` 等随意编号。

4. **检索为空时**：若 Sources 为空或仅含占位说明，正文先用一小段说明「基于当前索引未检索到相关文献片段」，并简要说明可能原因与建议（如重新 Ingest、调整关键词/文件名过滤），**不得**编造文献或脚注。
"""


def build_prompt(
    question,
    context,
    required_language="same as question",
    answer_style="哲学论述",
):
    style = normalize_answer_style(answer_style)

    style_block = f"""
Style mode: {style}
Write with maximal depth, strong conceptual architecture, and sustained argument.
Use fully developed paragraphs and avoid brief outline-like responses.
Differentiate analysis from mere summary; foreground tensions and conceptual commitments.
"""
    if style == STYLE_ACADEMIC:
        # DEPRECATED for frontend: 仍保留分支供旧 API 使用
        style_block = f"""
Style mode: {style} (deprecated in UI)
Write in a rigorous academic tone with explicit concepts and argument structure.
Prefer clarity and textual precision over rhetorical flourish.
"""
    elif style == STYLE_CONCISE:
        # DEPRECATED for frontend
        style_block = f"""
Style mode: {style} (deprecated in UI)
Be clear and focused while still preserving core argument steps and key footnotes.
Use shorter sections.
"""
    elif style == STYLE_PHILOSOPHER:
        style_block = f"""
Style mode: {style}
You compose as a philosopher writing for an expert audience: maximal conceptual depth,
sustained argument, explicit distinctions, and dialectical structure.
Avoid outline-style bullet substituting for real analysis; use full paragraphs.
"""
    elif style == STYLE_REVIEW:
        style_block = f"""
Style mode: {style}
You are an extremely strict dissertation blind reviewer.
Assume the user pasted a full draft section (possibly thousands of words) and wants severe quality control.
Tone: strict, unsparing, direct; but still constructive and professional.
Follow the same footnote convention as other modes when you quote their draft or attribute to corpus.
"""
    elif style == STYLE_CITE_PATCH:
        style_block = f"""
Style mode: {style}
The user's message in \"User input\" is a **draft to be annotated only**. Your job:

1. **Reproduce the user's draft verbatim** as the main body: **do not** change wording, order of paragraphs,
   punctuation for grammar, typos, or structure **except** inserting footnote markers and optional minimal
   clarifying brackets ONLY if absolutely necessary for disambiguation (prefer zero such edits; if in doubt, do not edit).

2. Where a claim or sentence can be supported by Sources, append after the relevant span the excerpt in parentheses
   followed by `[n]`, as: `（…verbatim or tight paraphrase from Sources…）[n]`. If the sentence already quotes the source,
   still add `[n]` after that parenthetical quote.

3. If a sentence **cannot** be tied to Sources, do **not** invent a footnote; you may append `[待核]` once per problematic
   sentence at most, without rewriting the sentence.

4. Then output `## 脚注` with each `[n]` matching the Cite as lines from Sources.

If Sources are empty, output only a short notice per global rules — do not fabricate annotated body.
"""
    elif style == STYLE_CONCEPT_MAP:
        style_block = f"""
Style mode: {style}
The user's input is **keyword-focused** (not necessarily a full question). You must:

1. Base the answer **primarily on direct quotation** from Sources; every major point should anchor to quoted lines
   followed by `[n]` in the required format.

2. If Sources overwhelmingly come from **one** document: explain how the concept(s) function **within that text**
   (definition, argumentative role, nearby concepts).

3. If Sources span **multiple** documents: contrast how the concept(s) differ across texts and, if appropriate,
   sketch a brief **conceptual genealogy** (who uses it how; tensions; lineage).

4. Do not substitute a general encyclopedia definition for textual analysis; state explicitly when Sources are thin.

5. Use the same `## 脚注` block at the end with full Cite as references.
"""
    elif style == STYLE_LITERATURE_REVIEW:
        style_block = f"""
Style mode: {style}
Write as a rigorous literature reviewer for an academic journal.
Organize the answer by research themes/controversies rather than by isolated author summaries.
For each theme, compare positions, evaluate evidence quality, and identify what remains unresolved.
Conclude with a synthetic assessment of research gaps and actionable future directions.
"""

    review_block = ""
    output_block = """
## Output Structure（哲学论述 / 默认）

1. 正文：若干完整段落，论证推进；引用一律 `（引文或转写）[n]`。
2. `## 脚注`：`[n]（…, p. …）` 逐条对应正文。
3. 末段可简要收束：限度、未决问题（如适用）。
"""

    if style == STYLE_REVIEW:
        review_block = """
## Review Mode（盲审审稿标准）

You are reviewing the user's own text as if it were a doctoral dissertation under blind review.
Be highly demanding; do NOT flatter.

Inspect: (1) thesis (2) logic (3) concepts (4) structure (5) evidence/citation (6) method (7) language (8) format.

For each major issue: brief quote or paraphrase from user draft → why it fails → concrete fix → optional rewrite.
Prioritize by severity.
When referring to corpus evidence in Sources, use the same `[n]` footnote convention.
"""
        output_block = """
## Output Structure（盲审审稿，强制）

1. Overall verdict（2–4 句）与主要风险。
2. High-severity（必改）。
3. Medium-severity。
4. Language / style。
5. Format / citation（对用户稿与对语料的引用均可用脚注）。
6. Prioritized revision plan。
7. 可选：改写好段落示例。
8. `## 脚注`：凡引用检查语料或用户稿外文献依据处，列出脚注。
"""

    elif style == STYLE_CITE_PATCH:
        output_block = """
## Output Structure（引文补注，强制）

1. **正文**：与用户来稿一致的完整文本，仅在支持处插入 `（引文）[n]`；禁止润色或重排。
2. `## 脚注`：与 `[n]` 一一对应。
"""

    elif style == STYLE_CONCEPT_MAP:
        output_block = """
## Output Structure（概念梳理，强制）

1. 开篇（不计入五节）：输入关键词与语料范围（单文本 / 多文本列表）。
2. **至少五节** `## 小标题`：术语界定与用法；论证位置；与其他概念关联；单文本内张力或**多文本对比**；**概念谱系 / 差异总括**。
3. 每节多个完整段落，密集 `（引文…）[n]`。
4. `## 脚注`：与正文编号一一对应，条数须满足上文「篇幅与覆盖面」。
"""
    elif style == STYLE_LITERATURE_REVIEW:
        output_block = """
## Output Structure（文献综述，强制）

1. `## 研究问题与综述范围`：界定问题、语料边界、判准。
2. `## 研究脉络与阶段`：按时期或问题域梳理发展线索。
3. `## 核心争论与立场比较`：逐项比较不同文献主张与分歧。
4. `## 方法、证据与论证质量评估`：说明各路径的优劣与局限。
5. `## 研究缺口与未来议题`：给出可执行的问题清单。
6. `## 脚注`：与正文编号一一对应，使用 Sources 的 Cite as。
"""

    elif style == STYLE_ACADEMIC:
        output_block = """
## Output Structure（学术分析，已弃用于前端）

与默认结构相同，语气更克制；脚注格式不变。
"""

    elif style == STYLE_CONCISE:
        output_block = """
## Output Structure（简洁作答，已弃用于前端）

较短正文 + `## 脚注`；不得省略脚注规范。
"""

    context_block = context.strip() if context else ""
    if not context_block:
        context_block = (
            "(No excerpts retrieved — 请按「检索为空」规则只输出简短说明，勿虚构文献。)"
        )

    footnote_block = _footnote_rules_block()
    length_block = _length_and_coverage_block(style)

    prompt = f"""
You are an elite philosophical researcher writing for publication.

Your primary task depends on Style mode below, using the provided Sources (excerpts) as evidence unless Sources are empty.

{style_block}

---

{length_block}

---

{footnote_block}

---

## Primary Objective

Use the provided excerpts as the main evidential basis. When excerpts are insufficient, state so clearly before any general knowledge.

---

## Evidence Priority Rule

1. Treat excerpts as primary evidence.
2. Do not attribute claims to texts unless clearly supported.
3. Never fabricate textual evidence or page numbers.
4. When excerpts are insufficient: distinguish (a) excerpt-supported from (b) general knowledge.

---

## Text Coverage Requirement

When multiple excerpts appear: synthesize where relevant; explain disagreements between sources explicitly.

---

## Comparative Interpretation

When texts disagree: compare positions and conceptual differences; do not conflate incompatible views.

---

{review_block}

---

## Language Rule

Match the user's language in \"User input\".
MANDATORY LANGUAGE OUTPUT: {required_language}

---

## Output Structure

{output_block}

---

Sources:
{context_block}

User input:
{question}

Answer:
"""
    return prompt
