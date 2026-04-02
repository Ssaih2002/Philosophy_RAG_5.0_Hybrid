import chromadb
from pathlib import Path
import shutil
from tqdm import tqdm

from .config import CHROMA_PATH


class VectorStore:
    def __init__(self, profile: str = "quality"):
        self.profile = profile
        base = Path(CHROMA_PATH)
        self.db_path = str(base.parent / f"{base.name}_{self.profile}")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection_name = f"philosophy_{self.profile}"
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            # 常见原因：用户手动删了部分 Chroma 文件（如 segment/bin），导致 sqlite 元数据与磁盘不一致。
            # 此时会报 NotFoundError: Collection [uuid] does not exist.
            msg = str(e)
            if "Collection" in msg and "does not exist" in msg:
                # 尝试清空并重建当前 profile 的 collection（保守做法：仅重建 collection，不动目录）
                try:
                    self.client.delete_collection(self.collection_name)
                except Exception:
                    pass
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                raise

    def reset_collection(self):
        """
        彻底重建当前 profile 的 Chroma 存储。

        仅 delete_collection 在某些“磁盘段文件已丢失/被删除”的情况下不足以恢复，
        会导致 query 时出现 NotFoundError: Collection [uuid] does not exist。
        因此这里直接清空整个持久化目录并重新创建 collection。
        """
        try:
            # 关闭/重建 client：不同版本 Chroma 对资源释放行为不同，这里直接重建最稳。
            self.client = None  # type: ignore[assignment]
        except Exception:
            pass
        try:
            shutil.rmtree(self.db_path, ignore_errors=True)
        except Exception:
            # 忽略：若文件被占用，后续会在 query/ingest 时抛更明确错误
            pass
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks, embeddings, batch_size=5000, show_progress=True):
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch")

        n = len(chunks)
        if n == 0:
            return

        batch_iter = range(0, n, batch_size)
        if show_progress:
            batch_iter = tqdm(
                batch_iter,
                total=(n + batch_size - 1) // batch_size,
                desc="写入 Chroma",
                unit="batch",
            )

        for start in batch_iter:
            end = min(start + batch_size, n)
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]

            ids = [f"chunk_{start + i}" for i in range(len(batch_chunks))]
            docs = [c["text"] for c in batch_chunks]
            metas = [
                {
                    "page": c["page"],
                    "source": c["source"],
                    "chunk_id": f"chunk_{start + i}",
                }
                for i, c in enumerate(batch_chunks)
            ]

            self.collection.add(
                ids=ids,
                documents=docs,
                embeddings=batch_embeddings,
                metadatas=metas,
            )

    def search(self, embedding, k, source_filters=None):
        kwargs = {}
        if source_filters:
            vals = [s for s in source_filters if s and str(s).strip()]
            if vals:
                kwargs["where"] = {"source": {"$in": vals}}
        try:
            return self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                **kwargs,
            )
        except Exception as e:
            # query 时仍可能遇到“collection uuid 不存在”的磁盘不一致问题
            msg = str(e)
            if "Collection" in msg and "does not exist" in msg:
                raise RuntimeError(
                    "Chroma 向量库已损坏或磁盘文件不一致（collection uuid 不存在）。"
                    "请先运行一次 Ingest 以重建向量库；必要时删除对应目录 data/chroma_db_<profile>/ 后再 Ingest。"
                ) from e
            raise