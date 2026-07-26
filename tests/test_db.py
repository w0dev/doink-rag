import sqlite3
import pytest
from rag.config import Settings
from rag.db import connect, create_schema, schema_dim


@pytest.fixture
def con(tmp_path):
    s = Settings(db_path=tmp_path / "t.db", embed_dim=8)
    c = connect(s)
    create_schema(c, s)
    return c


def test_schema_creates_all_tables(con):
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert {"documents", "chunks", "chunks_fts", "chunks_vec"} <= names


def test_schema_dim_reports_configured_dimension(con):
    assert schema_dim(con) == 8


def test_fts_trigger_mirrors_inserts(con):
    con.execute("INSERT INTO documents(id,path,content_hash,mtime,size) "
                "VALUES (1,'a.md','h',0,0)")
    con.execute("INSERT INTO chunks(id,document_id,ordinal,text,heading_path) "
                "VALUES (1,1,0,'the memory fit problem','Notes')")
    rows = con.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'memory'").fetchall()
    assert [r[0] for r in rows] == [1]


def test_fts_trigger_handles_delete(con):
    con.execute("INSERT INTO documents(id,path,content_hash,mtime,size) "
                "VALUES (1,'a.md','h',0,0)")
    con.execute("INSERT INTO chunks(id,document_id,ordinal,text,heading_path) "
                "VALUES (1,1,0,'swapping is bad','Notes')")
    con.execute("DELETE FROM chunks WHERE id=1")
    assert con.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'swapping'"
    ).fetchone()[0] == 0


def test_porter_stemming_is_active(con):
    con.execute("INSERT INTO documents(id,path,content_hash,mtime,size) "
                "VALUES (1,'a.md','h',0,0)")
    con.execute("INSERT INTO chunks(id,document_id,ordinal,text,heading_path) "
                "VALUES (1,1,0,'the process was swapping badly','Notes')")
    # 'swap' must match 'swapping'; this is the whole reason for porter over trigram
    assert con.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'swap'"
    ).fetchone()[0] == 1


def test_create_schema_rejects_dimension_mismatch(con, tmp_path):
    mismatched = Settings(db_path=tmp_path / "t.db", embed_dim=1024)
    with pytest.raises(RuntimeError, match="1024"):
        create_schema(con, mismatched)
    # the on-disk vec table must stay at its original dimension, not silently
    # drift to whatever the mismatched call asked for
    assert schema_dim(con) == 8


def test_create_schema_is_noop_when_dimension_matches(con, tmp_path):
    s = Settings(db_path=tmp_path / "t.db", embed_dim=8)
    create_schema(con, s)  # re-invoking with the same dimension must not raise
    assert schema_dim(con) == 8


def test_deleting_document_cascades_to_chunks(con):
    con.execute("INSERT INTO documents(id,path,content_hash,mtime,size) "
                "VALUES (1,'a.md','h',0,0)")
    con.execute("INSERT INTO chunks(id,document_id,ordinal,text,heading_path) "
                "VALUES (1,1,0,'x','y')")
    con.execute("DELETE FROM documents WHERE id=1")
    assert con.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
