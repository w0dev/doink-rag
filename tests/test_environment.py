"""The whole dense arm depends on loadable SQLite extensions.

pyenv-built Pythons on macOS ship without them, and the failure mode is a
confusing AttributeError deep inside db.connect(). Fail here instead, with a
message that says what to do.
"""
import sqlite3


def test_sqlite_supports_loadable_extensions():
    assert hasattr(sqlite3.Connection, "enable_load_extension"), (
        "This interpreter cannot load SQLite extensions, so sqlite-vec will not "
        "work. pyenv builds Python without them by default. Run under uv: "
        "`uv run pytest`."
    )
    con = sqlite3.connect(":memory:")
    con.enable_load_extension(True)


def test_fts5_with_porter_is_available():
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61')"
    )


def test_sqlite_vec_loads():
    import sqlite_vec

    con = sqlite3.connect(":memory:")
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    (version,) = con.execute("SELECT vec_version()").fetchone()
    assert version.startswith("v")
