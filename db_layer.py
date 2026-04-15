# -*- coding: utf-8 -*-
"""
Database layer for Ontology Database concept mapping tool
Handles SQLite operations and data retrieval
"""

import sqlite3
import logging
from typing import Any, Dict, Generator, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _str_list() -> List[str]:
    return []


@dataclass
class Concept:
    """Represents a concept/class from the ontology"""
    id: int
    ontology_id: str
    class_uri: str
    preferred_label: str
    definition: Optional[str] = None
    notation: Optional[str] = None
    obsolete: bool = False
    synonyms: List[str] = field(default_factory=_str_list)
    parent_uris: List[str] = field(default_factory=_str_list)


@dataclass
class OntologyInfo:
    """Represents ontology metadata"""
    id: str
    name: str
    num_classes: int
    status: str


class OntologyDB:
    """Manages SQLite database operations for ontology data"""

    def __init__(self, db_path: str = "bioportal.db"):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._validate_connection()
        logger.info(f"OntologyDB initialized with {db_path}")

    def _validate_connection(self):
        """Validate database connection and schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='classes'"
                )
                if not cursor.fetchone():
                    raise ValueError("Database schema not found. Ensure the database is populated.")
            logger.info("Database validation successful")
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            raise

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_ontologies(self) -> List[OntologyInfo]:
        """Retrieve all available ontologies"""
        query = "SELECT id, name, num_classes, status FROM ontologies ORDER BY name"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [
                OntologyInfo(
                    id=row["id"],
                    name=row["name"],
                    num_classes=row["num_classes"],
                    status=row["status"],
                )
                for row in cursor.fetchall()
            ]

    def search_by_label(
        self,
        query: str,
        ontology_ids: Optional[List[str]] = None,
        limit: int = 10,
        include_obsolete: bool = False,
    ) -> List[Concept]:
        """
        Search concepts by preferred label using FTS

        Args:
            query: Search text
            ontology_ids: Filter by specific ontologies
            limit: Maximum results
            include_obsolete: Include obsolete concepts

        Returns:
            List of matching concepts
        """
        sql = """
            SELECT DISTINCT c.id, c.ontology_id, c.class_uri,
                   c.preferred_label, c.definition, c.notation, c.obsolete
            FROM classes c
            WHERE c.preferred_label LIKE ?
        """
        params: List[Any] = [f"%{query}%"]

        if not include_obsolete:
            sql += " AND c.obsolete = 0"

        if ontology_ids:
            placeholders = ",".join("?" * len(ontology_ids))
            sql += f" AND c.ontology_id IN ({placeholders})"
            params.extend(ontology_ids)

        sql += " LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return [self._row_to_concept(row, conn) for row in cursor.fetchall()]

    def search_by_fts(
        self,
        query: str,
        ontology_ids: Optional[List[str]] = None,
        limit: int = 10,
        include_obsolete: bool = False,
    ) -> List[Concept]:
        """
        Full-text search using FTS5

        Args:
            query: Search text
            ontology_ids: Filter by specific ontologies
            limit: Maximum results
            include_obsolete: Include obsolete concepts

        Returns:
            List of matching concepts
        """
        sql = """
            SELECT DISTINCT c.id, c.ontology_id, c.class_uri,
                   c.preferred_label, c.definition, c.notation, c.obsolete
            FROM classes c
            WHERE (c.preferred_label LIKE ? OR c.definition LIKE ?)
        """
        params: List[Any] = [f"%{query}%", f"%{query}%"]

        if not include_obsolete:
            sql += " AND c.obsolete = 0"

        if ontology_ids:
            placeholders = ",".join("?" * len(ontology_ids))
            sql += f" AND c.ontology_id IN ({placeholders})"
            params.extend(ontology_ids)

        sql += " LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return [self._row_to_concept(row, conn) for row in cursor.fetchall()]

    def search_by_synonym(
        self,
        query: str,
        ontology_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Concept]:
        """
        Search by synonyms

        Args:
            query: Search text
            ontology_ids: Filter by specific ontologies
            limit: Maximum results

        Returns:
            List of matching concepts
        """
        sql = """
            SELECT DISTINCT c.id, c.ontology_id, c.class_uri,
                   c.preferred_label, c.definition, c.notation, c.obsolete
            FROM classes c
            JOIN synonyms s ON c.id = s.class_id
            WHERE s.synonym LIKE ?
        """
        params: List[Any] = [f"%{query}%"]

        if ontology_ids:
            placeholders = ",".join("?" * len(ontology_ids))
            sql += f" AND c.ontology_id IN ({placeholders})"
            params.extend(ontology_ids)

        sql += " LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return [self._row_to_concept(row, conn) for row in cursor.fetchall()]

    def get_by_uri(self, class_uri: str) -> Optional[Concept]:
        """Retrieve a concept by its URI"""
        query = """
            SELECT id, ontology_id, class_uri, preferred_label, definition, notation, obsolete
            FROM classes
            WHERE class_uri = ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (class_uri,))
            row = cursor.fetchone()
            return self._row_to_concept(row, conn) if row else None

    def _row_to_concept(self, row: sqlite3.Row, conn: sqlite3.Connection) -> Concept:
        """Convert database row to Concept object"""
        class_id = row["id"]

        syn_cursor = conn.cursor()
        syn_cursor.execute("SELECT synonym FROM synonyms WHERE class_id = ?", (class_id,))
        synonyms: List[str] = [str(s[0]) for s in syn_cursor.fetchall()]

        par_cursor = conn.cursor()
        par_cursor.execute("SELECT parent_uri FROM parents WHERE class_id = ?", (class_id,))
        parent_uris: List[str] = [str(p[0]) for p in par_cursor.fetchall()]

        return Concept(
            id=class_id,
            ontology_id=row["ontology_id"],
            class_uri=row["class_uri"],
            preferred_label=row["preferred_label"],
            definition=row["definition"],
            notation=row["notation"],
            obsolete=bool(row["obsolete"]),
            synonyms=synonyms,
            parent_uris=parent_uris,
        )

    @staticmethod
    def _fetch_in_batches(
        conn: sqlite3.Connection,
        query_template: str,
        ids: List[Any],
        chunk_size: int = 900,
    ) -> List[sqlite3.Row]:
        """Run a SELECT … WHERE x IN (?) query in chunks to stay within SQLite's
        variable limit (default 999).  query_template must contain exactly one
        '{ph}' placeholder that will be replaced with the right number of '?'s.
        """
        rows: List[sqlite3.Row] = []
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i : i + chunk_size]
            ph = ",".join("?" * len(chunk))
            cursor = conn.cursor()
            cursor.execute(query_template.format(ph=ph), chunk)
            rows.extend(cursor.fetchall())
        return rows

    def get_all_concepts_for_indexing(
        self, ontology_ids: Optional[List[str]] = None, batch_size: int = 1000
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Generator for retrieving all concepts with rich metadata for indexing.

        Uses batch JOIN queries instead of per-concept queries to avoid N+1
        performance issues.  For a batch of 1 000 concepts this runs 5 queries
        total instead of ~5 000.

        Includes: preferred_label, definition, synonyms, other labels,
                  parent labels, class_uri, ontology_id

        Args:
            ontology_ids: Optional filter by specific ontologies
            batch_size: Number of concepts per batch (keep ≤ 900 to avoid
                        hitting the SQLite variable limit for IN clauses)

        Yields:
            Batches of concept dicts with full enriched data
        """
        count_query = "SELECT COUNT(*) FROM classes"
        count_params: List[Any] = []
        if ontology_ids:
            placeholders = ",".join("?" * len(ontology_ids))
            count_query += f" WHERE ontology_id IN ({placeholders})"
            count_params.extend(ontology_ids)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(count_query, count_params)
            total: int = cursor.fetchone()[0]

        offset = 0
        while offset < total:
            batch_query = (
                "SELECT id, ontology_id, class_uri, preferred_label, definition, notation, obsolete "
                "FROM classes"
            )
            params: List[Any] = []

            if ontology_ids:
                placeholders = ",".join("?" * len(ontology_ids))
                batch_query += f" WHERE ontology_id IN ({placeholders})"
                params.extend(ontology_ids)

            batch_query += " LIMIT ? OFFSET ?"
            params.extend([batch_size, offset])

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(batch_query, params)
                rows = cursor.fetchall()

                if not rows:
                    break

                class_ids: List[int] = [row["id"] for row in rows]

                # ── Batch fetch synonyms (1 query for the whole batch) ──────
                syn_rows = self._fetch_in_batches(
                    conn,
                    "SELECT class_id, synonym FROM synonyms WHERE class_id IN ({ph})",
                    class_ids,
                )
                synonyms_map: Dict[int, List[str]] = {}
                for sr in syn_rows:
                    synonyms_map.setdefault(sr[0], []).append(str(sr[1]))

                # ── Batch fetch alternative labels ───────────────────────────
                lbl_rows = self._fetch_in_batches(
                    conn,
                    "SELECT class_id, label FROM labels WHERE class_id IN ({ph})",
                    class_ids,
                )
                labels_map: Dict[int, List[str]] = {}
                for lr in lbl_rows:
                    labels_map.setdefault(lr[0], []).append(str(lr[1]))

                # ── Batch fetch parent URIs ──────────────────────────────────
                par_rows = self._fetch_in_batches(
                    conn,
                    "SELECT class_id, parent_uri FROM parents WHERE class_id IN ({ph})",
                    class_ids,
                )
                parent_uris_map: Dict[int, List[str]] = {}
                for pr in par_rows:
                    parent_uris_map.setdefault(pr[0], []).append(str(pr[1]))

                # ── Batch fetch parent preferred labels ──────────────────────
                all_parent_uris: List[str] = list(
                    {uri for uris in parent_uris_map.values() for uri in uris}
                )
                parent_label_map: Dict[str, str] = {}
                if all_parent_uris:
                    plbl_rows = self._fetch_in_batches(
                        conn,
                        "SELECT class_uri, preferred_label FROM classes WHERE class_uri IN ({ph})",
                        all_parent_uris,
                    )
                    for plr in plbl_rows:
                        parent_label_map[str(plr[0])] = str(plr[1])

                # ── Assemble concept dicts ───────────────────────────────────
                concepts: List[Dict[str, Any]] = []
                for row in rows:
                    class_id: int = row["id"]
                    parent_uris: List[str] = parent_uris_map.get(class_id, [])
                    concepts.append(
                        {
                            "id": class_id,
                            "class_uri": str(row["class_uri"]),
                            "ontology_id": str(row["ontology_id"]),
                            "preferred_label": str(row["preferred_label"]),
                            "definition": row["definition"],
                            "notation": row["notation"],
                            "obsolete": bool(row["obsolete"]),
                            "synonyms": synonyms_map.get(class_id, []),
                            "labels": labels_map.get(class_id, []),
                            "parent_uris": parent_uris,
                            "parent_labels": [
                                parent_label_map[uri]
                                for uri in parent_uris
                                if uri in parent_label_map
                            ],
                        }
                    )

                yield concepts
            offset += batch_size

    def get_all_minimal_concepts(self) -> List[Dict[str, str]]:
        """Fast single-table scan returning only class_uri, preferred_label,
        ontology_id — no JOIN queries.  Used to rebuild the concepts-map cache
        when the retrieval indexes are already valid.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT class_uri, preferred_label, ontology_id FROM classes ORDER BY id"
            )
            return [
                {
                    "class_uri": str(row[0]),
                    "preferred_label": str(row[1]),
                    "ontology_id": str(row[2]),
                }
                for row in cursor.fetchall()
            ]

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ontologies")
            num_ontologies: int = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM classes")
            num_classes: int = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM synonyms")
            num_synonyms: int = cursor.fetchone()[0]

        return {
            "num_ontologies": num_ontologies,
            "num_classes": num_classes,
            "num_synonyms": num_synonyms,
        }
