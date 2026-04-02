import chromadb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
import networkx as nx
import re
from matplotlib import font_manager

from nltk.corpus import stopwords
from collections import Counter
from gensim import corpora
from gensim.models.ldamodel import LdaModel

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import jieba

nltk.download("stopwords")


def setup_plot_fonts():
    """
    在 Windows 上 matplotlib 经常默认字体不含中文，导致标签显示为空白/方块。
    这里自动挑一个可用的中文字体并设置为全局默认。
    """
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]

    chosen = None
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            chosen = name
            break
        except Exception:
            continue

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen]
    # 解决坐标轴负号显示为方块的问题
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def select_concepts_statistical(
    processed,
    *,
    min_df=20,
    max_df_ratio=0.3,
    top_n=3000,
):
    """
    统计式筛选“概念词”集合：
    - DF 过滤：去掉太稀有/太泛化的词
    - TF-IDF 近似打分：用全局 TF * IDF 排序，取 top_n
    """
    docs = [d for d in processed if d]
    n_docs = len(docs)
    if n_docs == 0:
        return set()

    df = Counter()
    tf = Counter()

    for doc in docs:
        tf.update(doc)
        df.update(set(doc))

    max_df = int(n_docs * max_df_ratio)
    if max_df < 1:
        max_df = 1

    candidates = [
        term
        for term, dfi in df.items()
        if dfi >= min_df and dfi <= max_df
    ]
    if not candidates:
        return set()

    import math

    scored = []
    for term in candidates:
        dfi = df[term]
        idf = math.log((n_docs + 1) / (dfi + 1)) + 1.0
        score = tf[term] * idf
        scored.append((score, term))

    scored.sort(reverse=True)
    return set(term for _, term in scored[:top_n])


def filter_processed_by_concepts(processed, concepts):
    if not concepts:
        return []
    filtered = []
    for doc in processed:
        kept = [t for t in doc if t in concepts]
        if kept:
            filtered.append(kept)
    return filtered

# -----------------------------
# 1 读取 ChromaDB
# -----------------------------

def load_chroma_documents():

    # 使用与向量库相同的路径和集合名，避免找不到集合导致越界
    client = chromadb.PersistentClient(path="data/chroma_db")

    # 如果不存在就创建，与 VectorStore 中保持一致
    collection = client.get_or_create_collection(
        name="philosophy",
        metadata={"hnsw:space": "cosine"}
    )

    # 分批读取，避免一次性取出太多文档导致 SQLite 报
    # "too many SQL variables"
    documents = []
    batch_size = 5000
    offset = 0

    while True:
        data = collection.get(limit=batch_size, offset=offset)
        batch_docs = data.get("documents", [])
        if not batch_docs:
            break
        documents.extend(batch_docs)
        offset += len(batch_docs)

    # 如果当前集合里还没有文档，给出清晰提示
    if not documents:
        raise RuntimeError(
            "ChromaDB 中当前集合 'philosophy' 为空，请先运行 ingest.py 构建向量库。"
        )

    return documents


# -----------------------------
# 2 文本预处理
# -----------------------------

def preprocess(texts):

    en_stop_words = set(stopwords.words("english"))


    zh_stop_words = {
        "的", "了", "和", "与", "及", "在", "对", "中", "为", "并",
    "是", "有", "也", "就", "而", "但", "被", "及其", "以及",
    "一个", "一种", "一些", "这个", "那个", "这些", "那些",
    "我们", "他们", "它们", "自己", "其", "此", "该",

    # 逻辑连接词
    "为了", "并且", "而且", "此外", "因此", "因而", "同时",
    "一方面", "另一方面", "如果", "那么", "虽然", "但是",
    "因为", "所以", "然而", "不过", "另外",

    # 情态词
    "可以", "可能", "必须", "应该", "能够", "需要",

    # 动态/结构词
    "已经", "正在", "仍然", "仍旧", "开始", "继续",
    "由于", "随着", "通过", "关于", "对于",

    # 常见虚词
    "并非", "并不", "不再", "不能", "没有",
    "以及其", "从而", "以便"
    }

    zh_noise_words = {
       "研究", "理论", "方法", "问题", "意义", "背景",
    "分析", "探讨", "讨论", "说明", "提出", "指出",
    "认为", "表明", "证明", "体现",

    "本文", "本书", "本研究", "本论文", "本章",
    "本节", "本部分", "本文认为",

    "首先", "其次", "再次", "最后",
    "进一步", "总体来看", "总的来说",

    "方面", "过程", "基础", "条件", "结果",
    "结构", "内容", "形式", "特征", "因素",

    "学界", "领域", "相关研究",

    "意义上", "层面", "维度",

     "出版社", "出版", "出版物", "出版年",
    "版", "年版", "印", "印刷",

    "参考文献", "参考", "文献",
    "目录", "摘要", "关键词",

    "第", "期", "卷", "辑",
    "年", "月", "日",

    "作者", "译者", "编者", "主编",
    "导言", "序言", "前言", "后记",

    "大学", "学院", "学报", "期刊",
    "论文", "硕士", "博士"
    }

    processed = []

    for text in texts:

        if not text:
            processed.append([])
            continue

        text = str(text)

        # 判断是否包含中文字符
        has_chinese = re.search(r"[\u4e00-\u9fff]", text) is not None

        tokens = []

        if has_chinese:
            # 中文：用 jieba 分词
            for tok in jieba.cut(text):
                tok = tok.strip()
                if not tok:
                    continue
                # 过滤标点和数字，保留长度>=2的中文词
                if (
                    re.fullmatch(r"[\u4e00-\u9fff]{2,}", tok)
                    and tok not in zh_stop_words
                    and tok not in zh_noise_words
                ):
                    # 再做一层规则过滤：出版信息/版本信息等
                    if "出版社" in tok:
                        continue
                    if tok.endswith(("出版社", "年版", "版")):
                        continue
                    tokens.append(tok)
        else:
            # 英文：原来的 token 逻辑，稍微再严格一点
            for word in text.split():
                word = word.lower()
                # 只保留纯字母且长度>=2的英文词
                if word.isalpha() and len(word) >= 2 and word not in en_stop_words:
                    tokens.append(word)

        processed.append(tokens)

    return processed


# -----------------------------
# 3 词频统计
# -----------------------------

def plot_word_frequency(processed):

    words = []

    for doc in processed:
        words.extend(doc)

    counter = Counter(words)

    common = counter.most_common(20)

    labels = [x[0] for x in common]
    values = [x[1] for x in common]

    plt.figure(figsize=(10,6))
    sns.barplot(x=values, y=labels)

    plt.title("Top Word Frequencies")
    plt.xlabel("Frequency")
    plt.ylabel("Concept")

    plt.show()


# -----------------------------
# 4 LDA主题模型
# -----------------------------

def run_lda(processed, topics=5):

    dictionary = corpora.Dictionary(processed)

    corpus = [dictionary.doc2bow(text) for text in processed]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=topics,
        passes=10
    )

    print("\nDetected Topics:\n")

    for idx, topic in lda.print_topics():

        print(f"Topic {idx}: {topic}")

    return lda, corpus, dictionary


# -----------------------------
# 5 主题演化曲线
# -----------------------------

def plot_topic_evolution(lda, corpus):

    topic_weights = []

    for doc in corpus:

        topics = lda.get_document_topics(doc)

        weights = [0]*lda.num_topics

        for topic_id, weight in topics:
            weights[topic_id] = weight

        topic_weights.append(weights)

    df = pd.DataFrame(topic_weights)

    plt.figure(figsize=(10,6))

    for i in range(df.shape[1]):
        plt.plot(df.index, df[i], label=f"Topic {i}")

    plt.title("Topic Evolution Across Text")
    plt.xlabel("Text Segment")
    plt.ylabel("Topic Weight")
    plt.legend()

    plt.show()


# -----------------------------
# 6 概念共现网络
# -----------------------------

def build_concept_network(
    processed,
    font_family=None,
    *,
    max_nodes=100,
    min_edge_weight=5
):

    window_size = 5

    G = nx.Graph()

    for doc in processed:

        for i in range(len(doc)):

            for j in range(i+1, min(i+window_size, len(doc))):

                w1 = doc[i]
                w2 = doc[j]

                if G.has_edge(w1, w2):

                    G[w1][w2]["weight"] += 1

                else:

                    G.add_edge(w1, w2, weight=1)

    if G.number_of_nodes() == 0:
        print("Concept network is empty; skip plotting.", flush=True)
        return

    # 只保留重要边（先粗筛）
    edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("weight", 0) > min_edge_weight]

    H = nx.Graph()
    for u, v, d in edges:
        H.add_edge(u, v, weight=d["weight"])

    if H.number_of_nodes() == 0:
        print("Concept network has no edges after filtering; skip plotting.", flush=True)
        return

    # 计算每个节点的重要性：加权度（边权之和）
    weighted_degree = {
        n: sum(attr.get("weight", 1) for _, _, attr in H.edges(n, data=True))
        for n in H.nodes()
    }

    # 只保留最关键的 max_nodes 个概念节点
    if max_nodes is not None and H.number_of_nodes() > max_nodes:
        keep_nodes = sorted(weighted_degree, key=weighted_degree.get, reverse=True)[:max_nodes]
        H = H.subgraph(keep_nodes).copy()

    plt.figure(figsize=(12, 12))

    pos = nx.spring_layout(H, k=0.8, seed=42)

    node_sizes = [max(80, min(1200, weighted_degree.get(n, 1) * 5)) for n in H.nodes()]
    # 连接线更细：只要能看见即可
    edge_widths = [0.25 for _ in H.edges()]

    nx.draw(
        H,
        pos,
        with_labels=True,
        node_size=node_sizes,
        width=edge_widths,
        font_size=9,
        font_family=font_family,
    )

    plt.title("Concept Co-occurrence Network")

    plt.show()


# -----------------------------
# 7 LDA交互可视化
# -----------------------------

def visualize_topics(lda, corpus, dictionary):

    vis = gensimvis.prepare(lda, corpus, dictionary)

    pyLDAvis.save_html(vis, "lda_topics.html")

    print("\nInteractive topic visualization saved to lda_topics.html\n")


# -----------------------------
# 主程序
# -----------------------------

def main():

    font_family = setup_plot_fonts()

    print("Loading texts from ChromaDB...", flush=True)

    texts = load_chroma_documents()

    print("Preprocessing texts...", flush=True)

    processed = preprocess(texts)

    print("Selecting concepts (statistical filtering)...", flush=True)
    concepts = select_concepts_statistical(
        processed,
        # 更激进的默认配置：更接近“哲学概念词表”
        min_df=50,
        max_df_ratio=0.15,
        top_n=1200,
    )
    processed_concepts = filter_processed_by_concepts(processed, concepts)
    print(
        f"Concept vocab size: {len(concepts)}; docs kept: {len(processed_concepts)}",
        flush=True
    )

    print("Generating word frequency statistics...", flush=True)

    plot_word_frequency(processed_concepts)

    print("Running LDA topic model...", flush=True)

    lda, corpus, dictionary = run_lda(processed_concepts)

    print("Plotting topic evolution...", flush=True)

    plot_topic_evolution(lda, corpus)

    print("Building concept network...", flush=True)

    build_concept_network(processed_concepts, font_family=font_family)

    print("Generating interactive topic visualization...", flush=True)

    visualize_topics(lda, corpus, dictionary)


if __name__ == "__main__":

    main()