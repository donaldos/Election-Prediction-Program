from __future__ import annotations

import logging

from models.chunk import ChunkWithEmbedding
from vectordb.base import AbstractVectorRepository, VectorRepositoryRegistry

logger = logging.getLogger(__name__)


@VectorRepositoryRegistry.register("pgvector")
class PgvectorRepository(AbstractVectorRepository):

    def __init__(
        self,
        collection: str = "election_chunks",
        dsn: str = "postgresql://localhost:5432/election_radar",
        dimensions: int = 1536,
    ) -> None:
        self._table_name = collection
        self._dsn = dsn
        self._dimensions = dimensions
        self._conn = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "pgvector"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        import psycopg
        from pgvector.psycopg import register_vector

        self._conn = psycopg.connect(self._dsn, autocommit=True)
        register_vector(self._conn)

        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id TEXT PRIMARY KEY,
                embedding vector({self._dimensions}),
                text TEXT,
                article_url TEXT,
                source TEXT,
                title TEXT,
                published_at TEXT,
                candidate TEXT,
                district_id TEXT,
                chunker_type TEXT
            )
        """)
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_embedding
            ON {self._table_name}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)

        self._loaded = True
        logger.info("[%s] 연결 완료 — table=%s", self.name, self._table_name)

    def _do_upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        sql = f"""
            INSERT INTO {self._table_name}
                (id, embedding, text, article_url, source, title, published_at, candidate, district_id, chunker_type)
            VALUES
                (%(id)s, %(embedding)s, %(text)s, %(article_url)s, %(source)s, %(title)s, %(published_at)s, %(candidate)s, %(district_id)s, %(chunker_type)s)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                text = EXCLUDED.text,
                article_url = EXCLUDED.article_url,
                source = EXCLUDED.source,
                title = EXCLUDED.title,
                published_at = EXCLUDED.published_at,
                candidate = EXCLUDED.candidate,
                district_id = EXCLUDED.district_id,
                chunker_type = EXCLUDED.chunker_type
        """
        for c in chunks:
            self._conn.execute(sql, {
                "id": c.id,
                "embedding": c.embedding,
                "text": c.text,
                "article_url": c.article_url,
                "source": c.source,
                "title": c.title,
                "published_at": str(c.published_at),
                "candidate": c.candidate,
                "district_id": c.district_id,
                "chunker_type": c.chunker_type,
            })
        return len(chunks)

    def _do_search(self, query_vector: list[float], top_k: int, filters: dict | None) -> list[dict]:
        where_clauses = []
        params: dict = {"query": query_vector, "limit": top_k}

        if filters:
            for i, (k, v) in enumerate(filters.items()):
                param_key = f"f{i}"
                where_clauses.append(f"{k} = %({param_key})s")
                params[param_key] = v

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT id, text, article_url, source, title, published_at,
                   candidate, district_id, chunker_type,
                   1 - (embedding <=> %(query)s::vector) AS score
            FROM {self._table_name}
            {where_sql}
            ORDER BY embedding <=> %(query)s::vector
            LIMIT %(limit)s
        """
        cur = self._conn.execute(sql, params)
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _do_delete(self, ids: list[str]) -> int:
        placeholders = ", ".join(["%s"] * len(ids))
        cur = self._conn.execute(
            f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})",
            ids,
        )
        return cur.rowcount

    def _do_count(self) -> int:
        cur = self._conn.execute(f"SELECT COUNT(*) FROM {self._table_name}")
        return cur.fetchone()[0]
