"""
SQL guard: both layers, including adversarial inputs. Offline; builds its own
analytics DB in a temp directory.
"""
import os
import sqlite3

import pytest

from core import sql_guard
from core.analytics_db import RESULTS_DIR, build_analytics_db, connect_readonly
from core.sql_guard import SQLGuardError, execute_guarded, validate_sql


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("guard")
    data_dir = tmp / "data"
    os.makedirs(data_dir)
    with sqlite3.connect(data_dir / "conflicts.db") as c:
        c.execute("""CREATE TABLE conflicts (id INTEGER PRIMARY KEY, timestamp TEXT,
            agent_a_name TEXT, agent_a_position TEXT, agent_a_reasoning TEXT,
            agent_b_name TEXT, agent_b_position TEXT, agent_b_reasoning TEXT,
            symbol TEXT, verdict TEXT, verdict_reasoning TEXT, resolved_by TEXT)""")
        c.execute("INSERT INTO conflicts VALUES (1,'2026-04-11T10:00:00','strategy','BUY AAPL',?,"
                  "'risk_guardian','VETO','r','AAPL','PENDING','p','orchestrator')", ("x" * 5000,))
    path = str(tmp / "analytics.db")
    build_analytics_db(path, results_dir=RESULTS_DIR, data_dir=str(data_dir))
    return path


@pytest.fixture
def conn(db_path):
    c = connect_readonly(db_path)
    yield c
    c.close()


def _code(sql):
    with pytest.raises(SQLGuardError) as e:
        validate_sql(sql)
    return e.value.code


# --- Legitimate analytics queries must pass ---------------------------------
@pytest.mark.parametrize("sql", [
    "SELECT sector, SUM(pnl) AS p FROM trades t JOIN symbols s ON s.symbol = t.symbol GROUP BY sector",
    "WITH m AS (SELECT exit_month, SUM(pnl) tp FROM trades_enriched GROUP BY exit_month) "
    "SELECT * FROM m ORDER BY tp DESC LIMIT 3",
    "SELECT strftime('%Y', exit_date) y, ROUND(AVG(pnl), 2) FROM trades GROUP BY y",
    "SELECT symbol, RANK() OVER (ORDER BY total_pnl DESC) r FROM perf_by_symbol",
    "SELECT CASE WHEN pnl > 0 THEN 'win' ELSE 'loss' END k, COUNT(*) FROM trades GROUP BY k",
    "SELECT e.date, e.equity, r.regime FROM equity_daily e JOIN market_regime_daily r USING (date)",
    "SELECT AVG(julianday(exit_date) - julianday(entry_date)) FROM trades",
    "SELECT symbol FROM trades WHERE pnl > 0 AND exit_reason IN ('TAKE_PROFIT') "
    "AND entry_date BETWEEN '2025-01-01' AND '2025-12-31' AND symbol LIKE 'A%'",
    "SELECT * FROM trades;",
])
def test_legitimate_queries_execute(conn, sql):
    result = execute_guarded(conn, sql)
    assert result.columns


# --- Statement-level attacks --------------------------------------------------
@pytest.mark.parametrize("sql,code", [
    ("DROP TABLE trades", "NOT_SELECT"),
    ("DELETE FROM trades", "NOT_SELECT"),
    ("UPDATE trades SET pnl = 0", "NOT_SELECT"),
    ("INSERT INTO symbols VALUES ('X', 'y')", "NOT_SELECT"),
    ("CREATE TABLE evil AS SELECT * FROM trades", "NOT_SELECT"),
    ("ALTER TABLE trades ADD COLUMN x", "NOT_SELECT"),
    ("PRAGMA table_info(trades)", "NOT_SELECT"),
    ("PRAGMA writable_schema = 1", "NOT_SELECT"),
    ("ATTACH DATABASE 'C:/evil.db' AS evil", "NOT_SELECT"),
    ("VACUUM INTO 'C:/copy.db'", "NOT_SELECT"),
    ("SELECT * INTO stolen FROM trades", "FORBIDDEN_STATEMENT"),
    # stacked queries, with and without comment camouflage
    ("SELECT 1; DROP TABLE trades", "MULTIPLE_STATEMENTS"),
    ("SELECT * FROM trades /* harmless */; DELETE FROM trades", "MULTIPLE_STATEMENTS"),
    ("SELECT 1 -- comment\n; DROP TABLE trades", "MULTIPLE_STATEMENTS"),
    ("SELECT 1;;ATTACH 'x.db' AS y", "MULTIPLE_STATEMENTS"),
])
def test_statement_attacks_are_blocked(sql, code):
    assert _code(sql) == code


# --- Data exfiltration --------------------------------------------------------
@pytest.mark.parametrize("sql", [
    "SELECT * FROM sqlite_master",
    "SELECT name FROM sqlite_schema",
    "SELECT sql FROM sqlite_temp_master",
    "SELECT symbol FROM trades UNION SELECT sql FROM sqlite_master",
    "SELECT symbol FROM trades UNION ALL SELECT name FROM sqlite_master WHERE type = 'table'",
    "SELECT * FROM trades WHERE symbol IN (SELECT name FROM sqlite_master)",
    "WITH x AS (SELECT * FROM sqlite_master) SELECT * FROM x",
    "SELECT * FROM main.trades",
    "SELECT * FROM temp.trades",
    "SELECT * FROM pragma_table_info('trades')",
    "SELECT/**/*/**/FROM/**/sqlite_master",
])
def test_disallowed_tables_are_blocked(sql):
    assert _code(sql) == "DISALLOWED_TABLE"


def test_unknown_column_is_blocked():
    assert _code("SELECT password FROM trades") == "DISALLOWED_COLUMN"


@pytest.mark.parametrize("sql", [
    "SELECT load_extension('evil.dll')",
    "SELECT readfile('C:/Windows/win.ini')",
    "SELECT writefile('x.txt', 'data')",
    "SELECT edit('x')",
    "SELECT fts3_tokenizer('simple')",
    "SELECT randomblob(1000000000)",
    "SELECT zeroblob(1000000000)",
    "SELECT random()",
])
def test_file_network_and_dos_functions_are_blocked(sql):
    assert _code(sql) == "DISALLOWED_FUNCTION"


# --- Limits ---------------------------------------------------------------------
def test_limit_is_forced_when_missing():
    assert "LIMIT 501" in validate_sql("SELECT * FROM trades")


def test_oversized_limit_is_reduced():
    assert "LIMIT 501" in validate_sql("SELECT * FROM equity_daily LIMIT 1000000")


def test_small_limit_is_kept():
    assert validate_sql("SELECT * FROM trades LIMIT 5").endswith("LIMIT 5")


def test_comments_are_stripped_from_executed_sql():
    assert "evil" not in validate_sql("SELECT symbol /* evil */ FROM trades -- evil")


def test_row_cap_reports_truncation(conn):
    result = execute_guarded(conn, "SELECT * FROM equity_daily", max_rows=10)
    assert len(result.rows) == 10 and result.hit_row_cap
    exact = execute_guarded(conn, "SELECT * FROM equity_daily LIMIT 10", max_rows=10)
    assert len(exact.rows) == 10 and not exact.hit_row_cap


def test_long_text_cells_are_truncated(conn):
    result = execute_guarded(conn, "SELECT agent_a_reasoning FROM agent_conflicts")
    assert len(result.rows[0][0]) <= sql_guard.MAX_CELL_CHARS + 1
    assert result.cells_truncated == 1


def test_slow_query_times_out(conn):
    with pytest.raises(SQLGuardError) as e:
        execute_guarded(conn, "SELECT COUNT(*) FROM equity_daily a, equity_daily b, equity_daily c",
                        timeout_s=0.05)
    assert e.value.code == "TIMEOUT"


def test_oversized_sql_is_rejected():
    assert _code("SELECT " + ", ".join(["pnl"] * 2000) + " FROM trades") == "TOO_LONG"


def test_garbage_is_a_parse_error():
    assert _code("SELEC FROM WHERE (((") in ("PARSE_ERROR", "NOT_SELECT")


# --- Layer 2 stands on its own --------------------------------------------------
@pytest.mark.parametrize("sql", [
    "SELECT * FROM sqlite_master",
    "PRAGMA table_info(trades)",
    "SELECT load_extension('x')",
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT * FROM r",
])
def test_authorizer_denies_even_if_layer_one_is_bypassed(conn, sql):
    """Run raw SQL under the authorizer only, as if the static check had a bug."""
    conn.set_authorizer(sql_guard._authorizer)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(sql).fetchall()
    finally:
        conn.set_authorizer(None)


def test_recursive_cte_is_refused_at_execution(conn):
    with pytest.raises(SQLGuardError) as e:
        execute_guarded(conn, "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) "
                              "SELECT * FROM r")
    assert e.value.code == "NOT_AUTHORIZED"


def test_handlers_are_cleared_after_a_query(conn):
    execute_guarded(conn, "SELECT COUNT(*) FROM trades")
    # With the authorizer gone, an internal read works again (proves no leak between calls).
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] > 0
